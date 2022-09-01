######################################################################
# Data Sourcing and Processing
# ----------------------------
#
# `torchtext library <https://pytorch.org/text/stable/>`__ has utilities for creating datasets that can be easily
# iterated through for the purposes of creating a language translation
# model. In this example, we show how to use torchtext's inbuilt datasets,
# tokenize a raw text sentence, build vocabulary, and numericalize tokens into tensor. We will use
# `Multi30k dataset from torchtext library <https://pytorch.org/text/stable/datasets.html#multi30k>`__
# that yields a pair of source-target raw sentences.
#
# To access torchtext datasets, please install torchdata following instructions at https://github.com/pytorch/data.
#

from torchtext.vocab import build_vocab_from_iterator
from typing import Iterable, List
import torch
import os
import numpy as np
import random
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']


class KjhRuDataset:
    def __init__(self,
                 data_root,
                 split='train',
                 language_pair=('kjh', 'ru'),
                 length=None):
        with open(f'{data_root}/{split}.{language_pair[0]}') as f:
            sents1 = f.readlines()

        with open(f'{data_root}/{split}.{language_pair[1]}') as f:
            sents2 = f.readlines()

        self.examples = [(sent1, sent2) for sent1, sent2 in zip(sents1, sents2)]
        self.length = length if length else len(self.examples)

    def __len__(self):
        return self.length

    def __iter__(self):
        for x in self.examples:
            yield x

    def __getitem__(self, i):
        i = i % len(self.examples)

        return self.examples[i]


######################################################################
# Collation
# ---------
#
# As seen in the ``Data Sourcing and Processing`` section, our data iterator yields a pair of raw strings.
# We need to convert these string pairs into the batched tensors that can be processed by our ``Seq2Seq`` network
# defined previously. Below we define our collate function that convert batch of raw strings into batch tensors that
# can be fed directly into our model.
#


# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input

    return func


# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))


def tokenize_text(text):
    """
    Tokenizes text from a string into a list of strings (tokens)
    """
    return text.split()


def prepare_comb_data(data_root_comb, language_pair_comb, min_freq):
    SRC_LANGUAGE, TGT_LANGUAGE = language_pair_comb

    # Place-holders
    token_transform = {}
    vocab_transform = {}

    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        token_transform[ln] = tokenize_text

    # helper function to yield list of tokens
    def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
        language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

        for data_sample in data_iter:
            yield tokenize_text(data_sample[language_index[language]])

    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        train_iter = KjhRuDataset(data_root_comb,
                                  split='train',
                                  language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
        # Create torchtext's Vocab object
        vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                        min_freq=min_freq,
                                                        specials=special_symbols,
                                                        special_first=True)

    # Set UNK_IDX as the default index. This index is returned when the token is not found.
    # If not set, it throws RuntimeError when the queried token is not found in the Vocabulary.
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        vocab_transform[ln].set_default_index(UNK_IDX)

    # src and tgt language text transforms to convert raw strings into tensors indices
    text_transform = {}
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        text_transform[ln] = sequential_transforms(token_transform[ln],  # Tokenization
                                                   vocab_transform[ln],  # Numericalization
                                                   tensor_transform)  # Add BOS/EOS and create tensor

    return token_transform, vocab_transform, text_transform


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def prepare_data(data_root_comb, language_pair_comb, data_root, language_pair, batch_size, min_freq, num_steps=None):
    token_transform_comb, vocab_transform_comb, text_transform_comb = prepare_comb_data(data_root_comb,
                                                                                        language_pair_comb,
                                                                                        min_freq)
    comb_ln2ln = {language_pair_comb[0]: language_pair[0],
                  language_pair_comb[1]: language_pair[1]}

    token_transform = {}
    vocab_transform = {}
    text_transform = {}
    for comb_ln, ln in comb_ln2ln.items():
        token_transform[ln] = token_transform_comb[comb_ln]
        vocab_transform[ln] = vocab_transform_comb[comb_ln]
        text_transform[ln] = text_transform_comb[comb_ln]

    SRC_LANGUAGE, TGT_LANGUAGE = language_pair

    # function to collate data samples into batch tensors
    def collate_fn(batch):
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip('\n')))
            tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip('\n')))

        src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
        tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
        return src_batch, tgt_batch

    g = torch.Generator()
    g.manual_seed(42)

    train_iter = KjhRuDataset(data_root,
                              split='train',
                              language_pair=(SRC_LANGUAGE, TGT_LANGUAGE),
                              length=num_steps)
    train_dataloader = DataLoader(train_iter, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
                                  num_workers=os.cpu_count(), worker_init_fn=seed_worker, generator=g)

    val_iter = KjhRuDataset(data_root,
                            split='val',
                            language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    val_dataloader = DataLoader(val_iter, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_dataloader, val_dataloader, vocab_transform, text_transform
