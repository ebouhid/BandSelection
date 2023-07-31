export EXP_NAME=RandomSeedTest
export EXP_COMP=pca
mkdir results/$EXP_NAME
nohup python genetic_alg.py 42 $EXP_COMP > results/$EXP_NAME/genetic_$EXP_COMP.out 2>&1 &
nohup python bruteforce.py 42 $EXP_COMP > results/$EXP_NAME/bruteforce_$EXP_COMP.out 2>&1 &
echo 'Job submitted'