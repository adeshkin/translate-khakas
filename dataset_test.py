from torchtext.vocab import build_vocab_from_iterator
from typing import Iterable, List
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


SRC_LANGUAGE = 'kjh'
TGT_LANGUAGE = 'ru'

# Place-holders
token_transform = {}
vocab_transform = {}


def tokenize(text):
    """
    Tokenizes text from a string into a list of strings (tokens)
    """
    return text.split(' ')


# helper function to yield list of tokens
def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

    for data_sample in data_iter:
        yield tokenize(data_sample[language_index[language]])


# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']


class KjhRuDataset:
    def __init__(self,
                 data_root='/home/adeshkin/projects/nmt/translate-khakas1/data/apply_bpe_kjh_kk_ru/kjh_ru',
                 split='test',
                 language_pair=('kjh', 'ru')):
        with open(f'{data_root}/{split}.{language_pair[0]}') as f:
            sents1 = [x.strip() for x in f.readlines()]

        with open(f'{data_root}/{split}.{language_pair[1]}') as f:
            sents2 = [x.strip() for x in f.readlines()]

        self.examples = [(sent1, sent2) for sent1, sent2 in zip(sents1, sents2)]

    def __len__(self):
        return len(self.examples)

    def __iter__(self):
        for x in self.examples:
            yield x


for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    # Training data Iterator
    train_iter = KjhRuDataset(split='test', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    # Create torchtext's Vocab object
    vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                    min_freq=1,
                                                    specials=special_symbols,
                                                    special_first=True)

# Set UNK_IDX as the default index. This index is returned when the token is not found.
# If not set, it throws RuntimeError when the queried token is not found in the Vocabulary.
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    vocab_transform[ln].set_default_index(UNK_IDX)

print(len(vocab_transform[ln]))
