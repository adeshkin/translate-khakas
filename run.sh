

for config in config/kjh_wmt19_kk_til_ky_ru/??/ru_??.*
do
  echo $config
  python main.py $config
done

for config in config/kjh_wmt19_kk_til_ky_ru/??/ru_???.*
do
  echo $config
  python main.py $config
done


