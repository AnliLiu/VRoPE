import math
import random
from data import ImageDetectionsField, TextField, RawField
from data import DataLoader, Sydney, UCM, RSICD, NWPU
import evaluation
from evaluation import PTBTokenizer, Cider
from models.transformer import Transformer, MemoryAugmentedEncoder, MeshedDecoder, ScaledDotProductAttentionMemory
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import NLLLoss
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse, os, pickle
import numpy as np
import itertools
import multiprocessing
from shutil import copyfile
from thop import profile
from fvcore.nn import FlopCountAnalysis
import warnings
warnings.filterwarnings("ignore")




def evaluate_loss(model, dataloader, loss_fn, text_field):
    # Validation loss
    model.eval()
    running_loss = .0
    with tqdm(desc='Epoch %d - validation' % e, unit='it', total=len(dataloader)) as pbar:
        with torch.no_grad():
            for it, (detections, detections_gl, rois, captions) in enumerate(
                    dataloader):  # batchsize是2, detection是10*50*2048, caption是10*15

                detections, detections_gl, rois, captions = detections.to(device), detections_gl.to(device), rois.to(
                    device), captions.to(
                    device)  # detection是10*50*2048, caption是10*15
                out = model(detections, detections_gl, rois, captions, isencoder=True)


                captions = captions[:, 1:].contiguous()
                out = out[:, :-1].contiguous()
                loss = loss_fn(out.view(-1, len(text_field.vocab)), captions.view(-1))
                this_loss = loss.item()
                running_loss += this_loss
                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()
    val_loss = running_loss / len(dataloader)
    return val_loss


def evaluate_metrics(model, dataloader, text_field):
    import itertools
    model.eval()
    gen = {}
    gts = {}
    with tqdm(desc='Epoch %d - evaluation' % e, unit='it', total=len(dataloader)) as pbar:
        for it, (images, caps_gt) in enumerate(dataloader):
            detections = images[0].to(device)
            detections_gl = images[1].to(device)
            ROIS= images[2].to(device)
            with torch.no_grad():
                out, _ = model.beam_search(detections, detections_gl ,ROIS, 20, text_field.vocab.stoi['<eos>'], 5, out_size=1)
            caps_gen = text_field.decode(out, join_words=False)
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen['%d_%d' % (it, i)] = [gen_i, ]
                gts['%d_%d' % (it, i)] = gts_i
            pbar.update()

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)
    return scores


def train_xe(model, dataloader, optim, text_field):
    # Training with cross-entropy
    model.train()
    scheduler.step()
    running_loss = .0
    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader)) as pbar:
        for it, (detections, detections_gl ,rois ,captions) in enumerate(
                dataloader):  # batchsize是2, detection是10*50*2048, caption是10*15

            detections, detections_gl, rois ,captions = detections.to(device), detections_gl.to(device), rois.to(device) ,captions.to(
                device)  # detection是10*50*2048, caption是10*15
            out = model(detections, detections_gl ,rois,captions ,isencoder=True)
            # isencoder = True
            # macs, params = profile(model, inputs=(detections, detections_gl, rois, captions , isencoder ))
            #
            # print('emacs = ' + str(macs*2 / 1000 ** 3) + 'G')
            # print('eParams = ' + str(params / 1000 ** 2) + 'M')
            optim.zero_grad()
            captions_gt = captions[:, 1:].contiguous()  # bs是10, captions是10，16；captions_gt是10,15；此时out是10,16,10199
            out = out[:, :-1].contiguous()  # 从10,16,10199变为 10,15,10199
            loss = loss_fn(out.view(-1, len(text_field.vocab)), captions_gt.view(-1))
            loss.backward()

            optim.step()
            this_loss = loss.item()
            running_loss += this_loss

            pbar.set_postfix(loss=running_loss / (it + 1))
            pbar.update()
            scheduler.step()

    loss = running_loss / len(dataloader)
    return loss


def train_scst(model, dataloader, optim, cider, text_field):
    # Training with self-critical
    tokenizer_pool = multiprocessing.Pool()
    running_reward = .0
    running_reward_baseline = .0
    model.train()
    running_loss = .0
    seq_len = 20
    beam_size = 2

    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader)) as pbar:
        for it, (images, caps_gt) in enumerate(dataloader):
            detections = images[0].to(device)
            detections_gl = images[1].to(device)
            ROIS= images[2].to(device)
            outs, log_probs = model.beam_search(detections, detections_gl,ROIS, 20, text_field.vocab.stoi['<eos>'],
                                                beam_size, out_size=beam_size)

            optim.zero_grad()

            # Rewards
            caps_gen = text_field.decode(outs.view(-1, seq_len))
            caps_gt = list(itertools.chain(*([c, ] * beam_size for c in caps_gt)))
            caps_gen, caps_gt = tokenizer_pool.map(evaluation.PTBTokenizer.tokenize, [caps_gen, caps_gt])
            reward = cider.compute_score(caps_gt, caps_gen)[1].astype(np.float32)
            reward = torch.from_numpy(reward).to(device).view(detections.shape[0], beam_size)
            reward_baseline = torch.mean(reward, -1, keepdim=True)
            loss = -torch.mean(log_probs, -1) * (reward - reward_baseline)

            loss = loss.mean()
            loss.backward()
            optim.step()

            running_loss += loss.item()
            running_reward += reward.mean().item()
            running_reward_baseline += reward_baseline.mean().item()
            pbar.set_postfix(loss=running_loss / (it + 1), reward=running_reward / (it + 1),
                             reward_baseline=running_reward_baseline / (it + 1))
            pbar.update()

    loss = running_loss / len(dataloader)
    reward = running_reward / len(dataloader)
    reward_baseline = running_reward_baseline / len(dataloader)
    return loss, reward, reward_baseline


def write_score(txt_path, scores, e):
    with open(txt_path, 'a') as f:
        f.write('........................................' + '\n')
        f.write('Iteration:' + str(e) + '\n')
        f.write('B1:' + str(scores['BLEU'][0]) + '\n')
        f.write('B2:' + str(scores['BLEU'][1]) + '\n')
        f.write('B3:' + str(scores['BLEU'][2]) + '\n')
        f.write('B4:' + str(scores['BLEU'][3]) + '\n')
        f.write('M:' + str(scores['METEOR']) + '\n')
        f.write('R:' + str(scores['ROUGE']) + '\n')
        f.write('C:' + str(scores['CIDEr']) + '\n')
        f.write('S:' + str(scores['SPICE']) + '\n')
        f.write('S*:' + str((scores['BLEU'][3] + scores['METEOR'] + scores['ROUGE'] + scores['CIDEr']) / 4) + '\n')
        f.write('Sm:' + str(
            (scores['BLEU'][3] + scores['METEOR'] + scores['ROUGE'] + scores['CIDEr'] + scores['SPICE']) / 5) + '\n')
        f.write('........................................' + '\n')
    f.close()


# flickr30k, train 145000, value 5070, test 5000
if __name__ == '__main__':
    import time

    seed = 14
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    time_start = time.time()  # 记录开始时间
    device = torch.device('cuda')
    # device = torch.cuda.set_device(0)
    parser = argparse.ArgumentParser(description='PKE-Transformer')
    parser.add_argument('--exp_name', type=str, default='NWPU')  # Sydney，UCM，RSICD
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--m', type=int, default=20)
    parser.add_argument('--head', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=10000)  # 1000
    parser.add_argument('--warm_up_epochs', type=int, default=10)  # 10
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--resume_last', action='store_true')
    parser.add_argument('--resume_best', action='store_true')  # coco_detections.hdf5

    ################################################################################################################
    # # # sydney
    parser.add_argument('--annotation_folder', type=str,
                        default='/media/liuli/pydata/pyproject/NWPU_Captions')
    parser.add_argument('--global_features_path', type=str,
                        default='/media/liuli/pydata/pyproject/data/NWPU/NWPU_res152_7_14')  # 196,2048
    parser.add_argument('--local_features_path', type=str,
                        default='/media/liuli/pydata/pyproject/data/NWPU/ROIS')  # 50*1024
    parser.add_argument('--rois_path', type=str,
                        default='/media/liuli/pydata/pyproject/data/NWPU/POS')  # 196,2048
    parser.add_argument('--txt_path', type=str, default='/media/liuli/pydata/pyproject/ROA/txt/roa_NWPU.txt')
    ################################################################################################################
    # ucm
    # parser.add_argument('--annotation_folder', type=str,
    #                     default='/media/liuli/pydata/pyproject/UCM_Captions')
    # parser.add_argument('--global_features_path', type=str,
    #                     default='/media/liuli/pydata/pyproject/data/UCM/UCM_res152_7_14')
    # parser.add_argument('--local_features_path', type=str,
    #                     default='/media/liuli/pydata/pyproject/data/UCM/ROIS')
    # parser.add_argument('--rois_path', type=str,
    #                     default='/media/liuli/pydata/pyproject/data/UCM/POS')  # 196,2048
    # parser.add_argument('--txt_path', type=str, default='/media/liuli/pydata/pyproject/ROA/ucm_result/GSA_NM.txt')
    # ################################################################################################################
    # rsicd
    #
    # parser.add_argument('--annotation_folder', type=str,
    #                     default='/media/dmd/ours/mlw/rs/RSICD_Captions')
    # parser.add_argument('--global_features_path', type=str,
    #                     default='/media/dmd/ours/mlw/rs/RSICD_Captions/global_feature/RSICD_res152_7_14')
    # parser.add_argument('--local_features_path', type=str,
    #                     default='/media/dmd/ours/mlw/rs/RSICD_Captions/local_feature/fine_res50/RSICD50')
    # parser.add_argument('--txt_path', type=str, default='2_rsicd_results/t1')

    ################################################################################################################


    ###############################    noise   gaussian   salt pepper  s&p  speckle ################################
    # gaussian_0_0.1    gaussian_3_0.01     GaussianBlur_3_0.5      GaussianBlur_5_0.5       GaussianBlur_5_1.5
    # parser.add_argument('--global_features_path', type=str,
    #                     default='/media/dmd/ours/mlw/rs/Sydney_Captions/noise/global/gaussian')
    # parser.add_argument('--local_features_path', type=str,
    #                     default='/media/dmd/ours/mlw/rs/Sydney_Captions/noise/local/gaussian')

    ############################################################################
    # parser.add_argument('--global_features_path', type=str,
    #                     default='/media/dmd/ours/mlw/rs/Sydney_Captions/global_feature/Sydney_res152_7_14')
    # parser.add_argument('--local_features_path', type=str,
    #                     default='/media/dmd/ours/mlw/rs/Sydney_Captions/noise/local/s&p')
    ############################################################################

    parser.add_argument('--logs_folder', type=str, default='tensorboard_logs')
    args = parser.parse_args()
    print(args)

    print('Meshed-Memory Transformer Training')
    # 日志
    writer = SummaryWriter(log_dir=os.path.join(args.logs_folder, args.exp_name))

    # Pipeline for image regions
    image_field = ImageDetectionsField(max_detections=32, load_in_tmp=False,
                                       global_detections_path=args.global_features_path,
                                       local_detections_path=args.local_features_path,
                                       rois_path=args.rois_path
                                       )

    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)

    # Create the dataset  Sydney，UCM，RSICD
    if args.exp_name == 'SYDEN':
        dataset = Sydney(image_field, text_field, 'Sydney/images/', args.annotation_folder, args.annotation_folder)
    elif args.exp_name == 'UCM':
        dataset = UCM(image_field, text_field, 'UCM/images/', args.annotation_folder, args.annotation_folder)
    elif args.exp_name == 'RSICD':
        dataset = RSICD(image_field, text_field, 'RSICD/images/', args.annotation_folder, args.annotation_folder)
    if args.exp_name == 'NWPU':
        dataset = NWPU(image_field, text_field, 'NWPU/images/', args.annotation_folder, args.annotation_folder)


    train_dataset, val_dataset, test_dataset = dataset.splits

    # 构建词汇表
    if not os.path.isfile('vocab_%s.pkl' % args.exp_name):
        print("Building vocabulary")
        text_field.build_vocab(train_dataset, val_dataset, min_freq=5)
        pickle.dump(text_field.vocab, open('vocab_%s.pkl' % args.exp_name, 'wb'))
    else:
        text_field.vocab = pickle.load(open('vocab_%s.pkl' % args.exp_name, 'rb'))

    # Model and dataloaders   ScaledDotProductAttentionMemory 即在K，V后面增加了memory. attention_module=None,则没有memory
    encoder = MemoryAugmentedEncoder(3, 0, attention_module=ScaledDotProductAttentionMemory,
                                     attention_module_kwargs={'m': args.m})
    decoder = MeshedDecoder(len(text_field.vocab), 127, 3, text_field.vocab.stoi['<pad>'])
    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)


    # 对train_dataset 处理， 构建 image检测特征与text字典  （调用image_dictionary）  image_field是之前建好的，RawField()是新建的
    dict_dataset_train = train_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    ref_caps_train = list(train_dataset.text)
    cider_train = Cider(PTBTokenizer.tokenize(ref_caps_train))
    dict_dataset_val = val_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField()})


    def lambda_lr(s):
        warm_up = args.warmup
        s += 1
        return (model.d_model ** -.5) * min(s ** -.5, s * warm_up ** -1.5)


    # warm_up_with_cos_lr = lambda epoch: epoch / args.warm_up_epochs if epoch <= args.warm_up_epochs else 0.5 * (
    #         math.cos((epoch - args.warm_up_epochs) / (args.epochs - args.warm_up_epochs) * math.pi) + 1)

    # Initial conditions
    # optim = Adam(model.parameters(), lr=1, betas=(0.9, 0.98))
    """local feature+slot最佳参数
    Sydneya1 0.87

    """
    optim = Adam(model.parameters(), lr=1, betas=(0.9, 0.98))
    # optim = Adam(model.parameters(), lr=0.78, betas=(0.9, 0.98))

    scheduler = LambdaLR(optim, lambda_lr)

    loss_fn = NLLLoss(ignore_index=text_field.vocab.stoi['<pad>'])
    use_rl = False
    # use_rl = True
    best_cider = .0
    patience = 0
    start_epoch = 0
    best_epoch = 0
    if args.resume_last or args.resume_best:
        if args.resume_last:
            fname = '/media/liuli/pydata/pyproject/ROA/result/N/%s_last.pth' % args.exp_name
        else:
            fname = '/media/liuli/pydata/pyproject/ROA/result/N/%s_best.pth' % args.exp_name

        if os.path.exists(fname):
            data = torch.load(fname)
            torch.set_rng_state(data['torch_rng_state'])
            torch.cuda.set_rng_state(data['cuda_rng_state'])
            np.random.set_state(data['numpy_rng_state'])
            random.setstate(data['random_rng_state'])
            model.load_state_dict(data['state_dict'], strict=False)
            optim.load_state_dict(data['optimizer'])
            scheduler.load_state_dict(data['scheduler'])
            start_epoch = data['epoch'] + 1
            best_cider = data['best_cider']
            patience = data['patience']
            use_rl = data['use_rl']
            print('Resuming from epoch %d, validation loss %f, and best cider %f' % (
                data['epoch'], data['val_loss'], data['best_cider']))

    print("Training starts")
    dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                  drop_last=True)
    dataloader_val = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    dict_dataloader_train = DataLoader(dict_dataset_train, batch_size=args.batch_size // 5, shuffle=True,
                                       num_workers=args.workers)
    dict_dataloader_val = DataLoader(dict_dataset_val, batch_size=args.batch_size // 5)
    dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.batch_size // 5)
    for e in range(start_epoch, start_epoch + 100):
        # train_loss = train_xe(model, dataloader_train, optim, text_field)
        # writer.add_scalar('data/train_loss', train_loss, e)

        if not use_rl:  # 不使用强化学习，则使用交叉熵损失
            train_loss = train_xe(model, dataloader_train, optim, text_field)
            writer.add_scalar('data/train_loss', train_loss, e)

        else:
            train_loss, reward, reward_baseline = train_scst(model, dict_dataloader_train, optim, cider_train,
                                                             text_field)
            writer.add_scalar('data/train_loss', train_loss, e)
            writer.add_scalar('data/reward', reward, e)
            writer.add_scalar('data/reward_baseline', reward_baseline, e)
            with open(args.txt_path, 'a') as f:
                f.write('****************************************' + '\n')
                f.write('Iteration:' + str(e) + '\n')
                f.write('reward:' + str(reward) + '\n')
                f.write('reward_baseline:' + str(reward_baseline) + '\n')
                f.write('****************************************' + '\n')
            f.close()

        # Validation loss
        val_loss = evaluate_loss(model, dataloader_val, loss_fn, text_field)
        writer.add_scalar('data/val_loss', val_loss, e)

        # Validation scores
        scores = evaluate_metrics(model, dict_dataloader_val, text_field)
        print("Validation scores", scores)
        val_cider = scores['CIDEr']
        writer.add_scalar('data/val_cider', val_cider, e)
        writer.add_scalar('data/val_bleu1', scores['BLEU'][0], e)
        writer.add_scalar('data/val_bleu2', scores['BLEU'][1], e)
        writer.add_scalar('data/val_bleu3', scores['BLEU'][2], e)
        writer.add_scalar('data/val_bleu4', scores['BLEU'][3], e)
        writer.add_scalar('data/val_meteor', scores['METEOR'], e)
        writer.add_scalar('data/val_rouge', scores['ROUGE'], e)
        writer.add_scalar('data/val_spice', scores['SPICE'], e)
        writer.add_scalar('data/val_S*',
                          (scores['BLEU'][3] + scores['METEOR'] + scores['ROUGE'] + scores['CIDEr']) / 4, e)
        writer.add_scalar('data/val_Sm', (
                scores['BLEU'][3] + scores['METEOR'] + scores['ROUGE'] + scores['CIDEr'] + scores['SPICE']) / 5, e)

        # Test scores
        scores = evaluate_metrics(model, dict_dataloader_test, text_field)
        print("Test scores", scores)
        test_cider = scores['CIDEr']
        writer.add_scalar('data/test_cider', scores['CIDEr'], e)
        writer.add_scalar('data/test_bleu1', scores['BLEU'][0], e)
        writer.add_scalar('data/test_bleu2', scores['BLEU'][1], e)
        writer.add_scalar('data/test_bleu3', scores['BLEU'][2], e)
        writer.add_scalar('data/test_bleu4', scores['BLEU'][3], e)
        writer.add_scalar('data/test_meteor', scores['METEOR'], e)
        writer.add_scalar('data/test_rouge', scores['ROUGE'], e)
        writer.add_scalar('data/test_spice', scores['SPICE'], e)
        writer.add_scalar('data/test_S*',
                          (scores['BLEU'][3] + scores['METEOR'] + scores['ROUGE'] + scores['CIDEr']) / 4, e)
        writer.add_scalar('data/test_Sm', (
                scores['BLEU'][3] + scores['METEOR'] + scores['ROUGE'] + scores['CIDEr'] + scores['SPICE']) / 5, e)

        # 写入txt
        write_score(args.txt_path, scores, e)

        # Prepare for next epoch
        best = False
        if test_cider >= best_cider:
            best_cider = test_cider
            patience = 0
            best = True
        else:
            patience += 1

        switch_to_rl = False
        exit_train = False
        if patience == 5:
            if not use_rl:
                use_rl = True
                switch_to_rl = True
                patience = 0
                # optim = Adam(model.parameters(), lr=5e-6)
                optim = Adam(model.parameters(), lr=5e-6)
                print("Switching to RL")
                with open(args.txt_path, 'a') as f:
                    f.write('////////////////////////////////////////' + '\n')
                    f.write('Switching to RL' + '\n')
                f.close()
            else:
                print('patience reached.')
                exit_train = True

        if switch_to_rl and not best:
            data = torch.load('/media/liuli/pydata/pyproject/ROA/result/N/%s_best.pth' % args.exp_name)
            torch.set_rng_state(data['torch_rng_state'])
            torch.cuda.set_rng_state(data['cuda_rng_state'])
            np.random.set_state(data['numpy_rng_state'])
            random.setstate(data['random_rng_state'])
            model.load_state_dict(data['state_dict'])
            print('Resuming from epoch %d, validation loss %f, and best cider %f' % (
                data['epoch'], data['val_loss'], data['best_cider']))

        torch.save({
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'random_rng_state': random.getstate(),
            'epoch': e,
            'val_loss': val_loss,
            'val_cider': val_cider,
            'state_dict': model.state_dict(),
            'optimizer': optim.state_dict(),
            'scheduler': scheduler.state_dict(),
            'patience': patience,
            'best_cider': best_cider,
            'use_rl': use_rl,
        }, '/media/liuli/pydata/pyproject/ROA/result/N/%s_last.pth' % args.exp_name)

        if best:
            copyfile('/media/liuli/pydata/pyproject/ROA/result/N/%s_last.pth' % args.exp_name, '/media/liuli/pydata/pyproject/ROA/result/N/%s_best.pth' % args.exp_name)
            best_epoch = e

        if exit_train:
            writer.close()
            break
        #
    time_end = time.time()  # 记录结束时间
    time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
    with open(args.txt_path, 'a') as f:
        f.write('////////////////////////////////////////' + '\n')
        f.write('time:' + str(time_sum)+ '\n')
        f.write('best_epoch:' + str(best_epoch)+ '\n')
        f.write('////////////////////////////////////////' + '\n')
    f.close()
    print(time_sum)
