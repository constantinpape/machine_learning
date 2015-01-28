#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

int main()
{
	std::fstream fs;

	fs.open("test", std::fstream::in);

// get length of file and line
	int count = 1;
	char aux[256];
	fs.getline(aux,256);
	std::stringstream ss(aux);

	std::cout << "AAA" << std::endl;
	int line_count = 0;
	while( !ss.eof() )
	{
		line_count++;
		char s[256];
		ss.get(s, 256);
		std::cout << line_count << " " << s << std::endl; 	
	}
	std::cout << "BBB" << std::endl;

	while( !fs.eof() )
	{
		count++;
		char s[256];
		fs.getline(s, 256);
		std::cout << count << "\n" << s << std::endl;
	}
	count--;
	std::cout << "Lines in file: " << count << std::endl; 
// get length of line
	
	using namespace boost::numeric::ublas;
	matrix<double> m(count,4);
	

	return 0;
}
