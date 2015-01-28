install : main.cpp common.o
	g++ -I ~/programming/boost/boost_1_57_0 -std=c++0x -o main main.cpp common.o

common.o : common.cpp common.h
	g++ -I ~/programming/boost/boost_1_57_0 -std=c++0x -c common.cpp

clean :
	rm *.o run
