
# train
#for config in config/dict_kjh_wmt19_thr_2_kk_ru/??/ru_??.*
#do
#  python main.py $config
#done

# finetune
for config in config/dict_kjh_wmt19_thr_2_kk_ru/??/ru_???.*
do
  python main.py $config
done

# train scratch kjh
for config in config/dict_kjh_wmt19_thr_2_kk_ru/???/*
do
  python main.py $config
done
