
def main():
    data_dir = 'apply_bpe_kjh_til_kk_ky_ru/kjh_ru'
    max_len = 0
    filepath = f'{data_dir}/train.kjh'
    with open(filepath) as f:
        sents = f.readlines()

    for sent in sents:
        if len(sent.split(' ')) > max_len:
            max_len = len(sent.split(' '))

    filepath = f'{data_dir}/train.ru'
    with open(filepath) as f:
        sents = f.readlines()

    for sent in sents:
        if len(sent.split(' ')) > max_len:
            max_len = len(sent.split(' '))

    print(max_len)


if __name__ == '__main__':
    main()
