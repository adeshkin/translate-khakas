import os
from nltk.tokenize import WordPunctTokenizer
from tqdm import tqdm
import random


def write_pairs(pairs, save_dir, src_lang, tgt_lang, split):
    os.makedirs(save_dir, exist_ok=True)
    src_filepath = f'{save_dir}/{split}.{src_lang}'
    tgt_filepath = f'{save_dir}/{split}.{tgt_lang}'
    with open(src_filepath, "w", encoding="utf-8") as src_f, \
            open(tgt_filepath, "w", encoding="utf-8") as tgt_f:
        for src_sent, tgt_sent in pairs:
            src_f.write(f"{src_sent.strip()}\n")
            tgt_f.write(f"{tgt_sent.strip()}\n")
            
            
def main():
    random.seed(42)
    data_dir = 'til_data'

    src_lang = 'kk'
    tgt_lang = 'ru'

    tokenizer = WordPunctTokenizer()
    save_dir = f'tok_data/til_{src_lang}_{tgt_lang}'
    for split in ['train', 'dev', 'test']:
        src_filepath = f'{data_dir}/{src_lang}-{tgt_lang}/{split}/{src_lang}-{tgt_lang}/{src_lang}-{tgt_lang}.{src_lang}'
        with open(src_filepath, 'r', encoding="utf-8") as src_f:
            src_sents = src_f.readlines()
        
        tgt_filepath = f'{data_dir}/{src_lang}-{tgt_lang}/{split}/{src_lang}-{tgt_lang}/{src_lang}-{tgt_lang}.{tgt_lang}'
        with open(tgt_filepath, 'r', encoding="utf-8") as tgt_f:
            tgt_sents = tgt_f.readlines()
        
        assert len(src_sents) == len(tgt_sents)
        pairs = []
        for src_sent, tgt_sent in tqdm(zip(src_sents, tgt_sents), total=len(src_sents)):
            src_sent = ' '.join(tokenizer.tokenize(src_sent.strip()))
            tgt_sent = ' '.join(tokenizer.tokenize(tgt_sent.strip()))

            pairs.append((src_sent, tgt_sent))

        random.shuffle(pairs)

        if split == 'dev':
            split = 'val'

        write_pairs(pairs, save_dir, src_lang, tgt_lang, split)


if __name__ == '__main__':
    main()