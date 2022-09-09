#!/bin/bash


source_dir=/home/adeshkin/projects/nmt/translate-khakas/data

num_operations=10000

sample='kjh_wmt19_thr_2_kk_ru'
src_lang='kjh_kk'
tgt_lang='ru'

tok_dir=$source_dir/tok_data/"$sample"
bpe_dir=$source_dir/learn_bpe/"$sample"

mkdir -p $bpe_dir

echo "learn_bpe.py ..."
subword-nmt learn-joint-bpe-and-vocab \
    --input $tok_dir/train."$src_lang" $tok_dir/train."$tgt_lang" \
    -s $num_operations \
    -o $bpe_dir/bpe.codes \
    --write-vocabulary $bpe_dir/bpe.vocab."$src_lang" $bpe_dir/bpe.vocab."$tgt_lang"