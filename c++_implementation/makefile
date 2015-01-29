# ~/programming/machine_learning/project/c++_implementation/source

install : test/test_readin.cpp test/test_bayes.cpp common.o BayesClassifier.o test_template.o
	g++ -I ~/programming/boost/boost_1_57_0  -std=c++0x -o bin/test-readin test/test_readin.cpp lib/common.o
	g++ -I ~/programming/boost/boost_1_57_0  -std=c++0x -o bin/test-bayes test/test_bayes.cpp lib/common.o lib/BayesClassifier.o lib/test_template.o

test_template.o : test/test_template.cpp common.o
	g++ -I ~/programming/boost/boost_1_57_0  -std=c++0x -c -o lib/test_template.o test/test_template.cpp lib/common.o

BayesClassifier.o : source/BayesClassifier.cpp source/BayesClassifier.h common.o
	g++ -I ~/programming/boost/boost_1_57_0 -std=c++0x -c -o lib/BayesClassifier.o source/BayesClassifier.cpp

common.o : source/common.cpp source/common.h
	g++ -I ~/programming/boost/boost_1_57_0 -std=c++0x -c -o lib/common.o source/common.cpp

clean :
	rm lib/*.o bin/*
