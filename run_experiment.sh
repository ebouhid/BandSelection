export EXP_NAME=$1
mkdir results/$EXP_NAME
nohup python genetic_alg.py $EXP_NAME 42 > results/$EXP_NAME/genetic.out 2>&1 &
nohup python bruteforce.py $EXP_NAME 42 > results/$EXP_NAME/bruteforce.out 2>&1 &
echo 'Job submitted'