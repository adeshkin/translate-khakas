
echo 'Training...'
python main.py config/kjh_kk_ru/kk_ru.yaml

echo 'Finetuning...'
python main.py config/kjh_kk_ru/kjh_ru.yaml