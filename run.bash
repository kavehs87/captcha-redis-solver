until python3 prod_load.py; do
  echo Daemon crashed, retrying in 10 seconds...
  sleep 10
done
