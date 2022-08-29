"""
Language Translation with nn.Transformer and torchtext
======================================================

This tutorial shows:
    - How to train a translation model from scratch using Transformer. 
    - Use tochtext library to access  `Multi30k <http://www.statmt.org/wmt16/multimodal-task.html#task1>`__ dataset to train a German to English translation model.
"""
######################################################################
# References
# ----------
#
# 1. Attention is all you need paper.
#    https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
# 2. The annotated transformer. https://nlp.seas.harvard.edu/2018/04/03/attention.html#positional-encoding

import os

import torch
import torch.nn as nn
from timeit import default_timer as timer
from tqdm import tqdm
import wandb
from torchtext.data.metrics import bleu_score
import math

from model import Seq2SeqTransformer
from utils import create_mask, translate
from dataset import prepare_data, PAD_IDX


def train_epoch(model, train_dataloader, device, loss_fn, optimizer):
    model.train()
    losses = 0

    for src, tgt in tqdm(train_dataloader):
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, device)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(train_dataloader)


def evaluate(model, val_dataloader, device, loss_fn):
    model.eval()
    losses = 0
    for src, tgt in tqdm(val_dataloader):
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, device)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(val_dataloader)


def main():
    project_name = 'nmt'
    experiment_name = 'default_transformer'
    save_dir = f'experiments/{project_name}/{experiment_name}'
    wandb_dir = f'{save_dir}/wandb_logs'
    os.makedirs(wandb_dir)
    checkpoint_dir = f'{save_dir}/checkpoints'
    os.makedirs(checkpoint_dir)

    wandb.init(project=project_name,
               name=experiment_name,
               dir=wandb_dir)

    SRC_DIR = '/home/adeshkin/projects/nmt/translate-khakas1/data/apply_bpe_kjh_kk_ru'
    DATA_ROOT_COMB = f'{SRC_DIR}/kjh_kk_ru'
    SRC_LANGUAGE_COMB = 'kjh_kk'
    TGT_LANGUAGE_COMB = 'ru'

    DATA_ROOT = f'{SRC_DIR}/kk_ru'
    SRC_LANGUAGE = 'kk'
    TGT_LANGUAGE = 'ru'

    MIN_FREQ = 1
    BATCH_SIZE = 64

    torch.manual_seed(0)

    train_dataloader, val_dataloader, vocab_transform, text_transform = prepare_data(DATA_ROOT_COMB,
                                                                                     (SRC_LANGUAGE_COMB,
                                                                                      TGT_LANGUAGE_COMB),
                                                                                     DATA_ROOT,
                                                                                     (SRC_LANGUAGE, TGT_LANGUAGE),
                                                                                     BATCH_SIZE,
                                                                                     MIN_FREQ)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    transformer = transformer.to(DEVICE)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    NUM_EPOCHS = 20
    best_val_loss = float('inf')

    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = timer()
        train_loss = train_epoch(transformer, train_dataloader, DEVICE, loss_fn, optimizer)
        end_time = timer()
        val_loss = evaluate(transformer, val_dataloader, DEVICE, loss_fn)
        wandb.log({'train_loss': train_loss,
                   'train_ppl': math.exp(train_loss),
                   'val_loss': val_loss,
                   'val_ppl': math.exp(val_loss)})

        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, "
               f"Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print('Saving best model...')
            torch.save(transformer.state_dict(), f"{checkpoint_dir}/best.pt")

    def calc_bleu(split='test'):
        src_filepath = f'{DATA_ROOT}/{split}.{SRC_LANGUAGE}'
        with open(src_filepath) as f:
            src_sents = f.readlines()

        tgt_filepath = f'{DATA_ROOT}/{split}.{TGT_LANGUAGE}'
        with open(tgt_filepath) as f:
            tgt_sents = f.readlines()

        assert len(src_sents) == len(tgt_sents)

        true_tgt_sents = []
        pred_tgt_sents = []
        examples = []
        for k, (src_sent, tgt_sent) in tqdm(enumerate(zip(src_sents, tgt_sents)), total=len(src_sents)):
            src_sent = src_sent.rstrip("\n")
            tgt_sent = tgt_sent.rstrip("\n")
            pred_tgt_sent = translate(transformer, src_sent, DEVICE, SRC_LANGUAGE, TGT_LANGUAGE, text_transform,
                                      vocab_transform)

            true_tgt_sent = tgt_sent.replace('@@ ', '')
            pred_tgt_sent = pred_tgt_sent + ' '
            pred_tgt_sent = pred_tgt_sent.replace('@@ ', '')

            pred_tgt_sents.append(pred_tgt_sent.split())
            true_tgt_sents.append([true_tgt_sent.split()])
            if k % 1000 == 1:
                examples.append((true_tgt_sent, pred_tgt_sent))

        return bleu_score(pred_tgt_sents, true_tgt_sents), examples

    print('Loading best model...')
    transformer.load_state_dict(torch.load(f'{checkpoint_dir}/best.pt'))
    test_bleu_score, test_examples = calc_bleu()

    test_metric_table = wandb.Table(columns=["bleu"])
    test_metric_table.add_data(test_bleu_score)
    wandb.log({"test_metrics": test_metric_table})

    prediction_table = wandb.Table(columns=["target", "prediction"])
    for example in test_examples:
        prediction_table.add_data(example[0], example[1])
    wandb.log({"test_translations": prediction_table})


if __name__ == '__main__':
    main()
