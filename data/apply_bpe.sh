#!/bin/bash


source_dir=/home/adeshkin/projects/nmt/translate-khakas/data

voc_thr=50

sample='dict_kjh_wmt19_thr_2_kk_ru'
src_lang='kjh_kk'
tgt_lang='ru'

sample1='dict_kjh_ru'
src_lang1='kjh'
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


