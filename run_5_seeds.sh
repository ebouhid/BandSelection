export EXP_NAME=NewOffspring
export EXP_COMP=pca

mkdir results/$EXP_NAME
nohup python genetic_alg.py $EXP_NAME 1 $EXP_COMP > results/$EXP_NAME/genetic_$EXP_COMP-1.out 2>&1 &
nohup python bruteforce.py $EXP_NAME 1 $EXP_COMP > results/$EXP_NAME/bruteforce_$EXP_COMP-1.out 2>&1 &

nohup python genetic_alg.py $EXP_NAME 10 $EXP_COMP > results/$EXP_NAME/genetic_$EXP_COMP-10.out 2>&1 &
nohup python bruteforce.py $EXP_NAME 10 $EXP_COMP > results/$EXP_NAME/bruteforce_$EXP_COMP-10.out 2>&1 &

nohup python genetic_alg.py $EXP_NAME 20 $EXP_COMP > results/$EXP_NAME/genetic_$EXP_COMP-20.out 2>&1 &
nohup python bruteforce.py $EXP_NAME 20 $EXP_COMP > results/$EXP_NAME/bruteforce_$EXP_COMP-20.out 2>&1 &

nohup python genetic_alg.py $EXP_NAME 30 $EXP_COMP > results/$EXP_NAME/genetic_$EXP_COMP-30.out 2>&1 &
nohup python bruteforce.py $EXP_NAME 30 $EXP_COMP > results/$EXP_NAME/bruteforce_$EXP_COMP-30.out 2>&1 &

nohup python genetic_alg.py $EXP_NAME 42 $EXP_COMP > results/$EXP_NAME/genetic_$EXP_COMP-42.out 2>&1 &
nohup python bruteforce.py $EXP_NAME 42 $EXP_COMP > results/$EXP_NAME/bruteforce_$EXP_COMP-42.out 2>&1 &

echo 'Job submitted'