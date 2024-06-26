export EXP_NAME=$1

mkdir results/$EXP_NAME
nohup python umda.py --exp-name $EXP_NAME --seed 1 > results/$EXP_NAME/genetic-1.out 2>&1 &
nohup python bruteforce.py $EXP_NAME 1 > results/$EXP_NAME/bruteforce-1.out 2>&1 &

nohup python umda.py --exp-name $EXP_NAME --seed 10 > results/$EXP_NAME/genetic-10.out 2>&1 &
nohup python bruteforce.py $EXP_NAME 10 > results/$EXP_NAME/bruteforce-10.out 2>&1 &

nohup python umda.py --exp-name $EXP_NAME --seed 20 > results/$EXP_NAME/genetic-20.out 2>&1 &
nohup python bruteforce.py $EXP_NAME 20 > results/$EXP_NAME/bruteforce-20.out 2>&1 &

nohup python umda.py --exp-name $EXP_NAME --seed 30 > results/$EXP_NAME/genetic-30.out 2>&1 &
nohup python bruteforce.py $EXP_NAME 30 > results/$EXP_NAME/bruteforce-30.out 2>&1 &

nohup python umda.py --exp-name $EXP_NAME --seed 42 > results/$EXP_NAME/genetic-42.out 2>&1 &
nohup python bruteforce.py $EXP_NAME 42 > results/$EXP_NAME/bruteforce-42.out 2>&1 &

echo 'Job submitted'