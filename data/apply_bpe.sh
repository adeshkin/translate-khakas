#!/bin/bash


source_dir=/home/adeshkin/projects/nmt/translate-khakas1/data

voc_thr=50

sample='kjh_til_kk_ky_ru'
src_lang='kjh_kk_ky'
tgt_lang='ru'

sample1='til_ky_ru'
src_lang1='ky'
tgt_lang1='ru'

bpe_dir=$source_dir/learn_bpe/"$sample"
tok_dir=$source_dir/tok_data/"$sample1"
bpe_result_dir=$source_dir/apply_bpe_"$sample"/"$sample1"
mkdir -p $bpe_result_dir

for mode in 'train' 'val' 'test'
do
    echo "apply_bpe.py to : ${src_lang1}.${mode}..."
    subword-nmt apply-bpe -c $bpe_dir/bpe.codes \
        --vocabulary $bpe_dir/bpe.vocab.$src_lang \
        --vocabulary-threshold $voc_thr < $tok_dir/$mode.$src_lang1 > $bpe_result_dir/$mode.$src_lang1
done

for mode in 'train' 'val' 'test'
do
    echo "apply_bpe.py to : ${tgt_lang1}.${mode}..."
    subword-nmt apply-bpe -c $bpe_dir/bpe.codes \
        --vocabulary $bpe_dir/bpe.vocab.$tgt_lang \
        --vocabulary-threshold $voc_thr < $tok_dir/$mode.$tgt_lang1 > $bpe_result_dir/$mode.$tgt_lang1
done


