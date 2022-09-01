import os
import random


def read_pairs(sample_dir, src_lang, tgt_lang, split):
    src_filepath = f'{sample_dir}/{split}.{src_lang}'
    with open(src_filepath, "r", encoding="utf-8") as f:
        src_sents = f.readlines()

    tgt_filepath = f'{sample_dir}/{split}.{tgt_lang}'
    with open(tgt_filepath, "r", encoding="utf-8") as f:
        tgt_sents = f.readlines()

    assert len(src_sents) == len(tgt_sents)
    pairs = [(src_sent, tgt_sent) for src_sent, tgt_sent in zip(src_sents, tgt_sents)]
    return pairs


def write_pairs(pairs, save_dir, src_lang, tgt_lang, split):
    os.makedirs(save_dir, exist_ok=True)
    src_filepath = f'{save_dir}/{split}.{src_lang}'
    tgt_filepath = f'{save_dir}/{split}.{tgt_lang}'
    with open(src_filepath, "w", encoding="utf-8") as src_f, \
            open(tgt_filepath, "w", encoding="utf-8") as tgt_f:
        for src_sent, tgt_sent in pairs:
            src_f.write(f"{src_sent.strip()}\n")
            tgt_f.write(f"{tgt_sent.strip()}\n")


def combine_data():
    random.seed(42)

    sample_dir1 = 'tok_data/kjh_til_kk_ru'
    src_lang1 = 'kjh_kk'
    tgt_lang1 = 'ru'

    sample_dir2 = 'tok_data/til_ky_ru'
    src_lang2 = 'ky'
    tgt_lang2 = 'ru'

    save_dir = 'tok_data/kjh_til_kk_ky_ru'
    src_lang = 'kjh_kk_ky'
    tgt_lang = 'ru'

    for split in ['train', 'val', 'test']:
        pairs1 = read_pairs(sample_dir=sample_dir1, src_lang=src_lang1, tgt_lang=tgt_lang1, split=split)
        pairs2 = read_pairs(sample_dir=sample_dir2, src_lang=src_lang2, tgt_lang=tgt_lang2, split=split)

        pairs = pairs1 + pairs2

        random.shuffle(pairs)
        
        print(len(pairs))
        print(pairs[:2])

        write_pairs(pairs, save_dir=save_dir, src_lang=src_lang, tgt_lang=tgt_lang, split=split)


if __name__ == '__main__':
    combine_data()
