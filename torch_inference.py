import torch
from nltk.tokenize import WordPunctTokenizer

from inference.custom_bpe import init_bpe
from data.dataset import tensor_transform, sequential_transforms, tokenize_text
from model import Seq2SeqTransformer
from utils import translate


def main():
    bpe_dir = 'data/learn_bpe/dict_kjh_wmt19_thr_2_kk_ru'
    voc_dir = 'data/apply_bpe_dict_kjh_wmt19_thr_2_kk_ru'
    model_path = 'experiments/nmt_wmt19_thr_2_kk/default_transformer_lang_comb_ru_kjh_kk_lang_ru_kjh_ft/checkpoints/best.pt'
    vocabulary_threshold = 50
    DEVICE = torch.device('cuda:0')
    SRC_LANGUAGE = 'ru'
    TGT_LANGUAGE = 'kjh'
    EMB_SIZE = 512
    NHEAD = 8
    FFN_HID_DIM = 512

    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3
    MAXLEN = 350

    bpe = init_bpe(bpe_dir, 'ru', vocabulary_threshold)

    language_pair_comb = (SRC_LANGUAGE, TGT_LANGUAGE)

    vocab_transform = dict()
    for lang in language_pair_comb:
        vocab_transform[lang] = torch.load(f'{voc_dir}/vocab_{lang}.pth')

    token_transform = dict()
    for ln in language_pair_comb:
        token_transform[ln] = tokenize_text

    text_transform = {}
    for ln in language_pair_comb:
        text_transform[ln] = sequential_transforms(token_transform[ln],
                                                   vocab_transform[ln],
                                                   tensor_transform)

    SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
    TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])

    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                     NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM, MAXLEN)
    transformer = transformer.to(DEVICE)

    transformer.load_state_dict(torch.load(model_path))

    input_sent = input(':\n')
    while input_sent != '#':
        input_sent = ' '.join(WordPunctTokenizer().tokenize(input_sent.strip()))
        src_sent = bpe.process_line(input_sent)
        print(src_sent)
        prd_sent_ = translate(transformer, src_sent, DEVICE, SRC_LANGUAGE, TGT_LANGUAGE, text_transform, vocab_transform)
        prd_sent = prd_sent_ + ' '
        print(prd_sent.replace('@@ ', '').strip())

        input_sent = input('\n:\n')


if __name__ == '__main__':
    main()
