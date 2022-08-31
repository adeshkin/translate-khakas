import torch
import random
from tqdm import tqdm


split = 'test'

src_filepath = f'{DATA_ROOT}/{split}.{SRC_LANGUAGE}'
with open(src_filepath) as src_f:
    src_sents = src_f.readlines()

tgt_filepath = f'{DATA_ROOT}/{split}.{TGT_LANGUAGE}'
with open(tgt_filepath) as tgt_f:
    tgt_sents = tgt_f.readlines()

assert len(src_sents) == len(tgt_sents)

true_tgt_sents = []
pred_tgt_sents = []
examples = []
example_idxs = [random.randint(0, len(src_sents) - 1) for _ in range(hparams['num_pred_examples'])]
for idx, (src_sent, tgt_sent) in tqdm(enumerate(zip(src_sents, tgt_sents)), total=len(src_sents), desc='TEST'):
    src_sent = src_sent.rstrip("\n")
    tgt_sent = tgt_sent.rstrip("\n")
    pred_tgt_sent = translate(transformer, src_sent, DEVICE, SRC_LANGUAGE, TGT_LANGUAGE, text_transform,
                              vocab_transform)

    true_tgt_sent = tgt_sent.replace('@@ ', '')
    pred_tgt_sent = pred_tgt_sent + ' '
    pred_tgt_sent = pred_tgt_sent.replace('@@ ', '')


transformer.load_state_dict(torch.load(f'{checkpoint_dir}/best.pt'))

