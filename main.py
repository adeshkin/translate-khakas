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
import sys
import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
from torchtext.data.metrics import bleu_score
import math
import random
import numpy as np

from model import Seq2SeqTransformer
from utils import create_mask, translate
from dataset import prepare_data, PAD_IDX


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train_epoch(model, train_dataloader, device, loss_fn, optimizer):
    model.train()
    losses = 0

    for src, tgt in tqdm(train_dataloader, desc='TRAIN'):
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
    with torch.no_grad():
        for src, tgt in tqdm(val_dataloader, desc='EVALUATE'):
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_input = tgt[:-1, :]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, device)

            logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

            tgt_out = tgt[1:, :]
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            losses += loss.item()

    return losses / len(val_dataloader)


def main(hparams):
    set_seed(42)
    project_name = hparams['project_name']
    MODEL_PATH = hparams['model_path']

    DATA_ROOT_COMB = hparams['data_root_comb']
    SRC_LANGUAGE_COMB = hparams['src_language_comb']
    TGT_LANGUAGE_COMB = hparams['tgt_language_comb']

    DATA_ROOT = hparams['data_root']
    SRC_LANGUAGE = hparams['src_language']
    TGT_LANGUAGE = hparams['tgt_language']

    experiment_name = f"{hparams['experiment_name']}_{SRC_LANGUAGE_COMB}_{TGT_LANGUAGE_COMB}_{SRC_LANGUAGE}_{TGT_LANGUAGE}"

    if hparams['add_info']:
        experiment_name += f"_{hparams['add_info']}"

    save_dir = f'experiments/{project_name}/{experiment_name}'
    wandb_dir = f'{save_dir}/wandb_logs'
    os.makedirs(wandb_dir)
    checkpoint_dir = f'{save_dir}/checkpoints'
    os.makedirs(checkpoint_dir)

    wandb.init(project=project_name,
               name=experiment_name,
               config=hparams,
               dir=wandb_dir)

    MIN_FREQ = hparams['min_freq']
    BATCH_SIZE = hparams['batch_size']
    NUM_STEPS = hparams['num_steps']

    train_dataloader, val_dataloader, vocab_transform, text_transform = prepare_data(DATA_ROOT_COMB,
                                                                                     (SRC_LANGUAGE_COMB,
                                                                                      TGT_LANGUAGE_COMB),
                                                                                     DATA_ROOT,
                                                                                     (SRC_LANGUAGE, TGT_LANGUAGE),
                                                                                     BATCH_SIZE,
                                                                                     MIN_FREQ,
                                                                                     NUM_STEPS)

    DEVICE = torch.device(hparams['device'])

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

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    transformer = transformer.to(DEVICE)

    if MODEL_PATH and os.path.exists(MODEL_PATH):
        print(f'Loading model from {MODEL_PATH}...')
        transformer.load_state_dict(torch.load(MODEL_PATH))

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    optimizer = torch.optim.Adam(transformer.parameters(), lr=hparams['lr'], betas=hparams['betas'], eps=hparams['eps'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           factor=hparams['factor'],
                                                           patience=hparams['patience'],
                                                           threshold=hparams['threshold'],
                                                           min_lr=hparams['min_lr'],
                                                           verbose=True)

    best_val_loss = float('inf')
    losses = 0
    num_steps_no_improv = 0
    transformer.train()
    optimizer.zero_grad()

    for step, (src, tgt) in enumerate(train_dataloader, start=1):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, DEVICE)

        logits = transformer(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()
        losses += loss.item()

        if step % int(hparams['check_val_every_n_steps'] / 10) == 0:
            os.system('clear')
            percentage = int(((step % hparams['check_val_every_n_steps']) / hparams['check_val_every_n_steps']) * 100)
            print(f'Step: {step}, Train:', '#' * percentage, f'{percentage} %')

        if step % hparams['num_accumulation_steps'] == 0:
            optimizer.step()
            optimizer.zero_grad()

        if step % hparams['check_val_every_n_steps'] == 0:
            train_loss = losses / hparams['check_val_every_n_steps']
            losses = 0

            val_loss = evaluate(transformer, val_dataloader, DEVICE, loss_fn)
            scheduler.step(val_loss)

            wandb.log({'lr': optimizer.param_groups[0]["lr"],
                       'train_loss': train_loss,
                       'train_ppl': math.exp(train_loss),
                       'val_loss': val_loss,
                       'val_ppl': math.exp(val_loss)})
            print(f"Step: {step}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}")

            if val_loss < best_val_loss:
                num_steps_no_improv = 0
                best_val_loss = val_loss
                print('Saving best model...')
                torch.save(transformer.state_dict(), f"{checkpoint_dir}/best.pt")
            else:
                num_steps_no_improv += 1

            if num_steps_no_improv == hparams['early_stop_patience']:
                print('Early stopping...')
                break
            transformer.train()
    # NUM_EPOCHS = hparams['num_epochs']
    # num_epochs_no_improv = 0
    # for epoch in range(1, NUM_EPOCHS + 1):
    #     train_loss = train_epoch(transformer, train_dataloader, DEVICE, loss_fn, optimizer)
    #     val_loss = evaluate(transformer, val_dataloader, DEVICE, loss_fn)
    #     scheduler.step(val_loss)
    #
    #     wandb.log({'lr': optimizer.param_groups[0]["lr"],
    #                'train_loss': train_loss,
    #                'train_ppl': math.exp(train_loss),
    #                'val_loss': val_loss,
    #                'val_ppl': math.exp(val_loss)})
    #
    #     print(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}")
    #
    #     if val_loss < best_val_loss:
    #         num_epochs_no_improv = 0
    #         best_val_loss = val_loss
    #         print('Saving best model...')
    #         torch.save(transformer.state_dict(), f"{checkpoint_dir}/best.pt")
    #     else:
    #         num_epochs_no_improv += 1
    #
    #     if num_epochs_no_improv == hparams['early_stop_patience']:
    #         print('Early stopping...')
    #         break

    def calc_bleu(split='test'):
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
            src_sent = src_sent.rstrip('\n')
            tgt_sent = tgt_sent.rstrip('\n')
            prd_sent = translate(transformer, src_sent, DEVICE, SRC_LANGUAGE, TGT_LANGUAGE, text_transform,
                                 vocab_transform)
            prd_sent = prd_sent.strip() + ' '

            norm_src_sent = src_sent.replace('@@ ', '')
            norm_tgt_sent = tgt_sent.replace('@@ ', '')

            norm_prd_sent = prd_sent.replace('@@ ', '')

            pred_tgt_sents.append(norm_prd_sent.split())
            true_tgt_sents.append([norm_tgt_sent.split()])
            if idx in example_idxs:
                examples.append((norm_src_sent, norm_tgt_sent, norm_prd_sent))

        return bleu_score(pred_tgt_sents, true_tgt_sents), examples

    print('Loading best model...')
    transformer.load_state_dict(torch.load(f'{checkpoint_dir}/best.pt'))
    test_bleu_score, test_examples = calc_bleu()

    test_metric_table = wandb.Table(columns=["bleu"])
    test_metric_table.add_data(test_bleu_score)
    wandb.log({"test_metrics": test_metric_table})

    prediction_table = wandb.Table(columns=["source", "target", "prediction"])
    for example in test_examples:
        prediction_table.add_data(example[0], example[1], example[2])
    wandb.log({"test_translations": prediction_table})


if __name__ == '__main__':
    filepath = sys.argv[1]
    with open(filepath, 'r') as f:
        hparams = yaml.load(f, yaml.Loader)
    main(hparams)
