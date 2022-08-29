#!/bin/bash


source_dir=/home/adeshkin/projects/nmt/translate-khakas1/data

num_operations=10000
voc_thr=50

sample='kjh_kk_ru'
k_lang='kjh_kk'

sample1='kjh_ru'
k_lang1='kjh'

sample2='kk_ru'
k_lang2='kk'

tok_dir=$source_dir/tok_data/"$sample"
bpe_dir=$source_dir/learn_bpe/"$sample"

mkdir -p $bpe_dir

echo "learn_bpe.py ..."
subword-nmt learn-joint-bpe-and-vocab \
    --input $tok_dir/train."$k_lang" $tok_dir/train.ru \
    -s $num_operations \
    -o $bpe_dir/bpe.codes \
    --write-vocabulary $bpe_dir/bpe.vocab."$k_lang" $bpe_dir/bpe.vocab.ru


tok_dir=$source_dir/tok_data/"$sample"
bpe_result_dir=$source_dir/apply_bpe_"$k_lang"/"$sample"
mkdir -p $bpe_result_dir

for lang in "$k_lang" 'ru'
do
  mode='train'
  echo "apply_bpe.py to : ${lang}.${mode}..."
  subword-nmt apply-bpe -c $bpe_dir/bpe.codes \
      --vocabulary $bpe_dir/bpe.vocab.$lang\
      --vocabulary-threshold $voc_thr < $tok_dir/$mode.$lang > $bpe_result_dir/$mode.$lang

done


tok_dir=$source_dir/tok_data/"$sample1"
bpe_result_dir=$source_dir/apply_bpe_"$k_lang"/"$sample1"
mkdir -p $bpe_result_dir
for lang in "$k_lang1" 'ru'
do
  bpe_lang=$lang
  if [[ $lang == "$k_lang1" ]]; then
      bpe_lang="$k_lang"
  fi
  for mode in 'train' 'val' 'test'
  do
      echo "apply_bpe.py to : ${lang}.${mode}..."
      subword-nmt apply-bpe -c $bpe_dir/bpe.codes \
          --vocabulary $bpe_dir/bpe.vocab.$bpe_lang\
          --vocabulary-threshold $voc_thr < $tok_dir/$mode.$lang > $bpe_result_dir/$mode.$lang
  done
done


tok_dir=$source_dir/tok_data/"$sample2"
bpe_result_dir=$source_dir/apply_bpe_"$k_lang"/"$sample2"
mkdir -p $bpe_result_dir
for lang in "$k_lang2" 'ru'
do
  bpe_lang=$lang
  if [[ $lang == "$k_lang2" ]]; then
      bpe_lang="$k_lang"
  fi
  for mode in 'train' 'val' 'test'
  do
      echo "apply_bpe.py to : ${lang}.${mode}..."
      subword-nmt apply-bpe -c $bpe_dir/bpe.codes \
          --vocabulary $bpe_dir/bpe.vocab.$bpe_lang\
          --vocabulary-threshold $voc_thr < $tok_dir/$mode.$lang > $bpe_result_dir/$mode.$lang
  done
done