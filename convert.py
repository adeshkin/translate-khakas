import torch
from model import Seq2SeqTransformer
from dataset import prepare_data
from utils import create_mask

SRC_DIR = '/home/adeshkin/projects/nmt/translate-khakas1/data/apply_bpe_kjh_kk_ru'
DATA_ROOT_COMB = f'{SRC_DIR}/kjh_kk_ru'
SRC_LANGUAGE_COMB = 'kjh_kk'
TGT_LANGUAGE_COMB = 'ru'

DATA_ROOT = f'{SRC_DIR}/kk_ru'
SRC_LANGUAGE = 'kk'
TGT_LANGUAGE = 'ru'

MIN_FREQ = 1
BATCH_SIZE = 1

torch.manual_seed(0)

train_dataloader, val_dataloader, vocab_transform, text_transform = prepare_data(DATA_ROOT_COMB,
                                                                                 (SRC_LANGUAGE_COMB,
                                                                                  TGT_LANGUAGE_COMB),
                                                                                 DATA_ROOT,
                                                                                 (SRC_LANGUAGE, TGT_LANGUAGE),
                                                                                 BATCH_SIZE,
                                                                                 MIN_FREQ)

DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
EMB_SIZE = 256
NHEAD = 8
FFN_HID_DIM = 512

NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
MAXLEN = 310

transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM, MAXLEN)

transformer.eval()
for src, tgt in val_dataloader:
    src = src.to(DEVICE)
    tgt = tgt.to(DEVICE)
    tgt_input = tgt[:-1, :]
    break

src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, DEVICE)

logits = transformer(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

torch.onnx.export(transformer)