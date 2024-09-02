import random
from data import ImageDetectionsField, TextField, RawField
from data import Sydney, UCM, RSICD, DataLoader
import evaluation
from models.transformer import Transformer, MemoryAugmentedEncoder, MeshedDecoder,ScaledDotProductAttentionMemory
import torch
from tqdm import tqdm
import argparse
import pickle
import numpy as np
import warnings
from thop import profile
warnings.filterwarnings("ignore")
random.seed(3)
torch.manual_seed(3)
np.random.seed(3)

def predict_captions(model, dataloader, text_field):
    import itertools
    model.eval()
    gen = {}
    gts = {}
    with tqdm(desc='evaluation', unit='it', total=len(dataloader)) as pbar:
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
    return scores,gen


if __name__ == '__main__':
    device = torch.device('cuda')
    parser = argparse.ArgumentParser(description='PKG-Transformer')
    parser.add_argument('--exp_name', type=str, default='UCM')
    # sydney
    parser.add_argument('--annotation_folder', type=str,
                        default='/media/liuli/pydata/pyproject/UCM_Captions')
    parser.add_argument('--global_features_path', type=str,
                        default='/media/liuli/pydata/pyproject/data/UCM/UCM_res152_7_14')
    parser.add_argument('--local_features_path', type=str,
                        default='/media/liuli/pydata/pyproject/data/UCM/ROIS')
    parser.add_argument('--rois_path', type=str,
                        default='/media/liuli/pydata/pyproject/data/UCM/POS')  # 196,2048
    parser.add_argument('--isss', type=str)

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--m', type=int, default=20)

    args = parser.parse_args()

    print('PKG-Transformer Evaluation')

    # Pipeline for image regions
    image_field = ImageDetectionsField(max_detections=50, load_in_tmp=False,
                                       global_detections_path=args.global_features_path,
                                       local_detections_path=args.local_features_path,
                                       rois_path=args.rois_path
                                       )

    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)

    # Create the dataset  Sydney，UCM，RSICD
    if args.exp_name == 'Sydney':
        dataset = Sydney(image_field, text_field, 'Sydney/images/', args.annotation_folder, args.annotation_folder)
    elif args.exp_name == 'UCM':
        dataset = UCM(image_field, text_field, 'UCM/images/', args.annotation_folder, args.annotation_folder)
    elif args.exp_name == 'RSICD':
        dataset = RSICD(image_field, text_field, 'RSICD/images/', args.annotation_folder, args.annotation_folder)

    train_dataset, val_dataset, test_dataset = dataset.splits

    text_field.vocab = pickle.load(open('vocab_%s.pkl' % args.exp_name, 'rb'))


    # Model and dataloaders

    encoder = MemoryAugmentedEncoder(3, 0, attention_module=ScaledDotProductAttentionMemory,
                                     attention_module_kwargs={'m': args.m})
    decoder = MeshedDecoder(len(text_field.vocab), 127, 3, text_field.vocab.stoi['<pad>'])
    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)

    data = torch.load('/media/liuli/pydata/pyproject/ROA/result/U/UCM_best.pth')

    model.load_state_dict(data['state_dict'])

    dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField()})

    dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.batch_size, num_workers=args.workers)

    scores,gen = predict_captions(model, dict_dataloader_test, text_field)

    with open('/media/liuli/pydata/pyproject/ROA/result/U/1.txt', 'a') as f:
        f.truncate(0)
        f.write('........................................' + '\n')
        f.write(str(gen))
        f.write('........................................' + '\n')
    f.close()
    print(scores)
