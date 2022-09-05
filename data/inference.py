import torch
import random
from tqdm import tqdm
import yaml

from dataset import prepare_data
from model import Seq2SeqTransformer
from utils import translate


filepath = 'config/kjh_ky_ru/ru_kjh.yaml'
with open(filepath, 'r') as f:
    hparams = yaml.load(f, yaml.Loader)



split = 'test'
SRC_LANGUAGE_COMB = hparams['src_language_comb']
TGT_LANGUAGE_COMB = hparams['tgt_language_comb']

lang = SRC_LANGUAGE_COMB if SRC_LANGUAGE_COMB != 'ru' else TGT_LANGUAGE_COMB
DATA_SRC_DIR = f"{hparams['data_src_dir']}/apply_bpe_{lang}_ru"
DATA_ROOT_COMB = f'{DATA_SRC_DIR}/{lang}_ru'


SRC_LANGUAGE = hparams['src_language']
TGT_LANGUAGE = hparams['tgt_language']
lang = SRC_LANGUAGE if SRC_LANGUAGE != 'ru' else TGT_LANGUAGE
DATA_ROOT = f'{DATA_SRC_DIR}/{lang}_ru'

DEVICE = torch.device(hparams['device'])

MIN_FREQ = hparams['min_freq']
BATCH_SIZE = hparams['batch_size']

train_dataloader, val_dataloader, vocab_transform, text_transform = prepare_data(DATA_ROOT_COMB,
                                                                                 (SRC_LANGUAGE_COMB,
                                                                                  TGT_LANGUAGE_COMB),
                                                                                 DATA_ROOT,
                                                                                 (SRC_LANGUAGE, TGT_LANGUAGE),
                                                                                 BATCH_SIZE,
                                                                                 MIN_FREQ)


project_name = hparams['project_name']
experiment_name = f"{hparams['experiment_name']}_{SRC_LANGUAGE_COMB}_{TGT_LANGUAGE_COMB}_{SRC_LANGUAGE}_{TGT_LANGUAGE}"

save_dir = f'experiments/{project_name}/{experiment_name}'

checkpoint_dir = f'{save_dir}/checkpoints'

SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
EMB_SIZE = hparams['EMB_SIZE']
NHEAD = hparams['NHEAD']
FFN_HID_DIM = hparams['FFN_HID_DIM']

NUM_ENCODER_LAYERS = hparams['NUM_ENCODER_LAYERS']
NUM_DECODER_LAYERS = hparams['NUM_DECODER_LAYERS']
MAXLEN = hparams['MAXLEN']

transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM, MAXLEN)
transformer = transformer.to(DEVICE)

transformer.load_state_dict(torch.load(f'{checkpoint_dir}/best.pt'))

# src_sent = ''
# while src_sent != '#':
#     src_sent = input(':')
#
#     pred_sent = translate(transformer, src_sent, DEVICE, SRC_LANGUAGE, TGT_LANGUAGE, text_transform, vocab_transform)
#     pred_sent = pred_sent.strip() + ' '
#     pred_sent = pred_sent.replace('@@ ', '')
#
#     print('src:', src_sent.replace('@@ ', ''))
#     print('pred:', pred_sent)
#     print()

src_filepath = f'{DATA_ROOT}/{split}.{SRC_LANGUAGE}'
with open(src_filepath) as src_f:
    src_sents = src_f.readlines()

tgt_filepath = f'{DATA_ROOT}/{split}.{TGT_LANGUAGE}'
with open(tgt_filepath) as tgt_f:
    tgt_sents = tgt_f.readlines()

assert len(src_sents) == len(tgt_sents)

pairs = []
for src_sent, tgt_sent in zip(src_sents, tgt_sents):
    pairs.append((src_sent.rstrip("\n"), tgt_sent.rstrip("\n")))

random.shuffle(pairs)

for i in range(10):
    src_sent = pairs[i][0]
    tgt_sent = pairs[i][1]
    prd_sent_ = translate(transformer, src_sent, DEVICE, SRC_LANGUAGE, TGT_LANGUAGE, text_transform, vocab_transform)
    prd_sent = prd_sent_ + ' '
    print('src:', src_sent.replace('@@ ', ''))
    print('tgt:', tgt_sent.replace('@@ ', ''))
    print('prd:', prd_sent.replace('@@ ', ''))
    print(prd_sent_.replace('@@ ', '').split())
    print()



