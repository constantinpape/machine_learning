# ~/programming/machine_learning/project/c++_implementation/source

install : test/test_readin.cpp common.o
	g++ -I ~/programming/boost/boost_1_57_0  -std=c++0x -o bin/test-readin test/test_readin.cpp lib/common.o

common.o : source/common.cpp source/common.h
	g++ -I ~/programming/boost/boost_1_57_0 -std=c++0x -c -o lib/common.o source/common.cpp

clean :
	rm lib/*.o bin/*
