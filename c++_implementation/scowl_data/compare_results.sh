#! /bin/sh
#compare classification and generation results 
usage="
usage: $0 [target_file]"

if [ $# -ne 1 ]; then
	echo $usage
	exit 1
fi

python2 mk_data.py
data=$(basename $1)
#data="l${length}.70"
classifier="densitytree bayes copula"

# not need using arrays
# clsfy=(densitytree bayes copula)
# for c in ${clsfy[@]} ; do

for c in $classifier; do
	../bin/test_$c original/data_train.out original/labels_train.out original/data_test.out test_$c/$data
	echo
	echo $(date) $data >> ../results/$c.txt 
	python2 eval.py original/labels_test.out test_$c/$data >> ../results/$c.txt
done


vimdiff ../results/*
