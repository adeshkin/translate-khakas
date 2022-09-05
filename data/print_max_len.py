
def main():
    data_dir = 'apply_bpe_dict_kjh_wmt19_thr_2_kk_ru/dict_kjh_wmt19_thr_2_kk_ru'
    max_len = 0
    src_lang = 'kjh_kk'
    tgt_lang = 'ru'
    for split in ['train', 'val', 'test']:
        filepath = f'{data_dir}/{split}.{src_lang}'
        with open(filepath) as f:
            sents = f.readlines()

        for sent in sents:
            if len(sent.split(' ')) > max_len:
                max_len = len(sent.split())

        filepath = f'{data_dir}/{split}.{tgt_lang}'
        with open(filepath) as f:
            sents = f.readlines()

        for sent in sents:
            if len(sent.split(' ')) > max_len:
                max_len = len(sent.split())

    print(max_len)


if __name__ == '__main__':
    main()
