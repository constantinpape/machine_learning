# ~/programming/machine_learning/project/c++_implementation/source

install : build test/test_readin.cpp test/test_sortindex.cpp test/test_bayes.cpp test/test_densitytree.cpp test/test_copula.cpp
	g++ -I ~/programming/boost/boost_1_57_0  -std=c++0x -o bin/test-readin test/test_readin.cpp lib/common.o
	g++ -I ~/programming/boost/boost_1_57_0  -std=c++0x -o bin/test-sortindex test/test_sortindex.cpp lib/common.o
	g++ -I ~/programming/boost/boost_1_57_0  -std=c++0x -o bin/test-bayes test/test_bayes.cpp lib/BayesClassifier.o lib/test_template.o lib/common.o
	g++ -I ~/programming/boost/boost_1_57_0  -std=c++0x -o bin/test-densitytree test/test_densitytree.cpp lib/DensityTreeClassifier.o lib/test_template.o lib/common.o lib/node_t.o
	g++ -I ~/programming/boost/boost_1_57_0  -std=c++0x -o bin/test-copula test/test_copula.cpp lib/DensityTreeClassifier.o lib/BayesClassifier.o lib/CopulaClassifier.o lib/test_template.o lib/common.o lib/node_t.o

build : BayesClassifier.o DensityTreeClassifier.o CopulaClassifier.o test_template.o common.o node_t.o

test_template.o : test/test_template.cpp
	g++ -I ~/programming/boost/boost_1_57_0  -std=c++0x -c -o lib/test_template.o test/test_template.cpp

CopulaClassifier.o : source/CopulaClassifier.cpp source/CopulaClassifier.h
	g++ -I ~/programming/boost/boost_1_57_0 -std=c++0x -c -o lib/CopulaClassifier.o source/CopulaClassifier.cpp

DensityTreeClassifier.o : source/DensityTreeClassifier.cpp source/DensityTreeClassifier.h
	g++ -I ~/programming/boost/boost_1_57_0 -std=c++0x -c -o lib/DensityTreeClassifier.o source/DensityTreeClassifier.cpp

BayesClassifier.o : source/BayesClassifier.cpp source/BayesClassifier.h
	g++ -I ~/programming/boost/boost_1_57_0 -std=c++0x -c -o lib/BayesClassifier.o source/BayesClassifier.cpp

node_t.o : source/node_t.cpp source/node_t.h 
	g++ -I ~/programming/boost/boost_1_57_0 -std=c++0x -c -o lib/node_t.o source/node_t.cpp

common.o : source/common.cpp source/common.h
	g++ -I ~/programming/boost/boost_1_57_0 -std=c++0x -c -o lib/common.o source/common.cpp

clean :
	rm lib/*.o bin/*
