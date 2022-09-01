



for config in config/kjh_ky_ru/ru_kjh*
do
  echo $config
  python main.py $config
done


for config in config/kjh_ky_ru/ru_ky*
do
  echo $config
  python main.py $config
done