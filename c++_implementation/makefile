# ~/programming/machine_learning/project/c++_implementation/source

install : build classifier_tests

test_depth : build test/test_depth.cpp
	g++ -I ~/programming/boost/boost_1_57_0  -std=c++0x -o bin/test-depth test/test_depth.cpp lib/DensityTreeClassifier.o lib/BayesClassifier.o lib/CopulaClassifier.o lib/common.o lib/node_t.o

test_nearestneighbors : build test/test_nearestneighbors.cpp
	g++ -I ~/programming/boost/boost_1_57_0  -std=c++0x -o bin/test-nearestneighbors test/test_nearestneighbors.cpp lib/DensityTreeClassifier.o lib/BayesClassifier.o lib/CopulaClassifier.o lib/common.o lib/node_t.o

test_readin : test/test_readin.cpp 
	g++ -I ~/programming/boost/boost_1_57_0  -std=c++0x -o bin/test-readin test/test_readin.cpp lib/common.o

test_sortindex: test/test_sortindex.cpp
	g++ -I ~/programming/boost/boost_1_57_0  -std=c++0x -o bin/test-sortindex test/test_sortindex.cpp lib/common.o

classifier_tests : test/test_bayes.cpp test/test_densitytree.cpp test/test_copula.cpp test/test_densityforest.cpp test/test_copulaforest.cpp
	g++ -I ~/programming/boost/boost_1_57_0  -std=c++0x -o bin/test-bayes test/test_bayes.cpp lib/BayesClassifier.o lib/test_template.o lib/common.o
	g++ -I ~/programming/boost/boost_1_57_0  -std=c++0x -o bin/test-densitytree test/test_densitytree.cpp lib/DensityTreeClassifier.o lib/test_template.o lib/common.o lib/node_t.o
	g++ -I ~/programming/boost/boost_1_57_0  -std=c++0x -o bin/test-copula test/test_copula.cpp lib/DensityTreeClassifier.o lib/BayesClassifier.o lib/CopulaClassifier.o lib/test_template.o lib/common.o lib/node_t.o
	g++ -I ~/programming/boost/boost_1_57_0  -std=c++0x -o bin/test-densityforest test/test_densityforest.cpp lib/DensityForestClassifier.o lib/test_template.o lib/common.o lib/node_t.o lib/DensityTreeClassifier.o
	g++ -I ~/programming/boost/boost_1_57_0  -std=c++0x -o bin/test-copulaforest test/test_copulaforest.cpp lib/DensityForestClassifier.o lib/BayesClassifier.o lib/DensityTreeClassifier.o lib/CopulaForestClassifier.o lib/test_template.o lib/common.o lib/node_t.o

build : BayesClassifier.o DensityTreeClassifier.o CopulaClassifier.o DensityForestClassifier.o CopulaForestClassifier.o test_template.o common.o node_t.o

test_template.o : test/test_template.cpp
	g++ -I ~/programming/boost/boost_1_57_0  -std=c++0x -c -o lib/test_template.o test/test_template.cpp

CopulaForestClassifier.o : source/CopulaForestClassifier.cpp source/CopulaForestClassifier.h
	g++ -I ~/programming/boost/boost_1_57_0 -std=c++0x -c -o lib/CopulaForestClassifier.o source/CopulaForestClassifier.cpp

DensityForestClassifier.o : source/DensityForestClassifier.cpp source/DensityForestClassifier.h
	g++ -I ~/programming/boost/boost_1_57_0 -std=c++0x -c -o lib/DensityForestClassifier.o source/DensityForestClassifier.cpp

CopulaClassifier.o : source/CopulaClassifier.cpp source/CopulaClassifier.h
	g++ -I ~/programming/boost/boost_1_57_0 -std=c++0x -c -o lib/CopulaClassifier.o source/CopulaClassifier.cpp

DensityTreeClassifier.o : source/DensityTreeClassifier.cpp source/DensityTreeClassifier.h
	g++ -O3 -I ~/programming/boost/boost_1_57_0 -std=c++0x -c -o lib/DensityTreeClassifier.o source/DensityTreeClassifier.cpp

BayesClassifier.o : source/BayesClassifier.cpp source/BayesClassifier.h
	g++ -I ~/programming/boost/boost_1_57_0 -std=c++0x -c -o lib/BayesClassifier.o source/BayesClassifier.cpp

node_t.o : source/node_t.cpp source/node_t.h 
	g++ -I ~/programming/boost/boost_1_57_0 -std=c++0x -c -o lib/node_t.o source/node_t.cpp

common.o : source/common.cpp source/common.h
	g++ -I ~/programming/boost/boost_1_57_0 -std=c++0x -c -o lib/common.o source/common.cpp

clean :
	rm lib/*.o bin/*
