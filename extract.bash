for i in `seq 50 69`;
do
  echo worker $i
  # on cloud:
  #xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python extract.py $i &
  #xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python extract.py $i &
  # on macbook for debugging:
  python extract.py $i &
  sleep 1.0
done
