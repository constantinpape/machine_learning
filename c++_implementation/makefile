# ~/programming/machine_learning/project/c++_implementation/source

install : test/test_readin.cpp test/test_bayes.cpp BayesClassifier.o test_template.o
	g++ -I ~/programming/boost/boost_1_57_0  -std=c++0x -o bin/test-readin test/test_readin.cpp lib/common.o
	g++ -I ~/programming/boost/boost_1_57_0  -std=c++0x -o bin/test-bayes test/test_bayes.cpp lib/BayesClassifier.o lib/test_template.o

build : common.o node_t.o BayesClassifier.o DensityTreeClassifier.o test_template.o

test_template.o : test/test_template.cpp common.o
	g++ -I ~/programming/boost/boost_1_57_0  -std=c++0x -c -o lib/test_template.o test/test_template.cpp lib/common.o

DensityTreeClassifier.o : source/DensityTreeClassifier.cpp source/DensityTreeClassifier.h node_t.o
	g++ -I ~/programming/boost/boost_1_57_0 -std=c++0x -c -o lib/DensityTreeClassifier.o source/DensityTreeClassifier.cpp lib/node_t.o

BayesClassifier.o : source/BayesClassifier.cpp source/BayesClassifier.h
	g++ -I ~/programming/boost/boost_1_57_0 -std=c++0x -c -o lib/BayesClassifier.o source/BayesClassifier.cpp

node_t.o : source/node_t.cpp source/node_t.h 
	g++ -I ~/programming/boost/boost_1_57_0 -std=c++0x -c -o lib/node_t.o source/node_t.cpp

common.o : source/common.cpp source/common.h
	g++ -I ~/programming/boost/boost_1_57_0 -std=c++0x -c -o lib/common.o source/common.cpp

clean :
	rm lib/*.o bin/*
