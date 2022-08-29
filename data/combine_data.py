import os
import random


def read_pairs(sample_dir, k_lang):
    k_filepath = f'{sample_dir}/train.{k_lang}'
    with open(k_filepath, "r", encoding="utf-8") as f:
        k_sents = f.readlines()

    r_filepath = f'{sample_dir}/train.ru'
    with open(r_filepath, "r", encoding="utf-8") as f:
        r_sents = f.readlines()

    assert len(k_sents) == len(r_sents)
    pairs = [(k_sent, r_sent) for k_sent, r_sent in zip(k_sents, r_sents)]
    return pairs


def write_pairs(pairs, save_dir, k_lang):
    os.makedirs(save_dir)
    k_filepath = f'{save_dir}/train.{k_lang}'
    r_filepath = f'{save_dir}/train.ru'
    with open(k_filepath, "w", encoding="utf-8") as k_f, \
            open(r_filepath, "w", encoding="utf-8") as r_f:
        for k_sent, r_sent in pairs:
            k_f.write(f"{k_sent.strip()}\n")
            r_f.write(f"{r_sent.strip()}\n")


def combine_data():
    data_dir = 'tok_data'

    kjh_ru_pairs = read_pairs(sample_dir=f'{data_dir}/kjh_ru', k_lang='kjh')
    kk_ru_pairs = read_pairs(sample_dir=f'{data_dir}/kk_ru', k_lang='kk')
    ky_ru_pairs = read_pairs(sample_dir=f'{data_dir}/ky_ru', k_lang='ky')
    source_kk_ru_pairs = read_pairs(sample_dir=f'{data_dir}/source_kk_ru', k_lang='kk')

    kk_kjh_pairs = kjh_ru_pairs + kk_ru_pairs
    ky_kjh_pairs = kjh_ru_pairs + ky_ru_pairs
    source_kk_kjh_pairs = kjh_ru_pairs + source_kk_ru_pairs

    print(len(kjh_ru_pairs), len(kk_ru_pairs), len(ky_ru_pairs), len(source_kk_ru_pairs))
    print(len(kk_kjh_pairs), len(ky_kjh_pairs), len(source_kk_kjh_pairs))

    random.seed(42)
    random.shuffle(kk_kjh_pairs)
    random.seed(42)
    random.shuffle(ky_kjh_pairs)
    random.seed(42)
    random.shuffle(source_kk_kjh_pairs)

    print(kk_kjh_pairs[:2])
    print(ky_kjh_pairs[:2])
    print(source_kk_kjh_pairs[:2])

    write_pairs(kk_kjh_pairs, save_dir=f'{data_dir}/kjh_kk_ru', k_lang='kjh_kk')
    write_pairs(ky_kjh_pairs, save_dir=f'{data_dir}/kjh_ky_ru', k_lang='kjh_ky')
    write_pairs(source_kk_kjh_pairs, save_dir=f'{data_dir}/kjh_source_kk_ru', k_lang='kjh_kk')


if __name__ == '__main__':
    combine_data()