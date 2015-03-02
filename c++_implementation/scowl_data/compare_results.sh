#! /bin/sh
#compare classification and generation results 
length=7
data="l${length}.70"
classifier="densitytree bayes copula"

# not need using arrays
# clsfy=(densitytree bayes copula)
# for c in ${clsfy[@]} ; do

for c in $classifier; do
	../bin/test_$c original/data_train.out original/labels_train.out original/data_test.out test_$c/$data
	echo $(date) $data > ../results/$c.txt 
	python2 eval.py original/labels_test.out test_$c/$data >> ../results/$c.txt
done

#python2 eval.py original/labels_test.out test_bayes/$data >> ../results/bayes.txt
#python2 eval.py original/labels_test.out test_dt/$data >> ../results/dt.txt
#python2 eval.py original/labels_test.out test_copula/$data >> ../results/copula.txt


vimdiff ../results/*
