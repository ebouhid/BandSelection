export EXP_NAME=CO_mut_probs
export EXP_COMP=pca
mkdir results/$EXP_NAME
nohup python genetic_alg.py > results/$EXP_NAME/genetic_$EXP_COMP.out 2>&1 &
nohup python bruteforce.py > results/$EXP_NAME/bruteforce_$EXP_COMP.out 2>&1 &