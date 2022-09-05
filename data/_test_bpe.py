from nltk.tokenize import WordPunctTokenizer
import torch



from dataset import UNK_IDX, sequential_transforms, tensor_transform, tokenize_text
from apply_bpe import init_bpe

bpe_dir = 'data/learn_bpe/kjh_wmt19_kk_til_ky_ru'
lang = 'ru'
vocabulary_threshold = 50

bpe = init_bpe(bpe_dir, lang, vocabulary_threshold)

data_root_comb = 'data/apply_bpe_kjh_wmt19_kk_til_ky_ru/kjh_wmt19_kk_til_ky_ru'
language_pair_comb = ('ru', 'kjh_kk_ky')
min_freq = 1
_, vocab_transform, _ = prepare_comb_data(data_root_comb,
                                          language_pair_comb,
                                          min_freq)
# for lang in vocab_transform_comb:
#     print(type(vocab_transform_comb[lang]))
#     torch.save(vocab_transform_comb[lang], f'vocab_{lang}.pth')
#
# vocab_transform = dict()
# for lang in language_pair_comb:
#     vocab_transform[lang] = torch.load(f'vocab_{lang}.pth')

for ln in language_pair_comb:
    vocab_transform[ln].set_default_index(UNK_IDX)

token_transform = dict()
for ln in language_pair_comb:
    token_transform[ln] = tokenize_text

text_transform = {}
for ln in language_pair_comb:
    text_transform[ln] = sequential_transforms(token_transform[ln],
                                               vocab_transform[ln],
                                               tensor_transform)
LANGUAGE = 'ru'

input_sent = input(':\n')
while input_sent != '#':
    input_sent = ' '.join(WordPunctTokenizer().tokenize(input_sent.strip()))
    input_sent_ = bpe.process_line(input_sent)
    print(input_sent_)
    sample = text_transform[LANGUAGE](input_sent_)
    print(sample)
    input_sent = input(':\n')

# tensor([    2,   368,    15, 12425,  2586,  3974,     4,    25,   443,   947,
#            39,   776,     6,  7863,  3404,   324,  1334,    55,   250,  1905,
#          1034,     3])
