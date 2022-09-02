
# train
for config in config/kjh_wmt19_kk_til_ky_ru/??/ru_??.*
do
  python main.py $config
done

# finetune
for config in config/kjh_wmt19_kk_til_ky_ru/??/ru_???.*
do
  python main.py $config
done

# train scratch kjh
for config in config/kjh_wmt19_kk_til_ky_ru/???/*
do
  python main.py $config
done
