

for config in config/*/??_*
do
  echo $config
  python main.py $config
done

for config in config/*/???_*
do
  echo $config
  python main.py $config
done