#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cstdlib>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <algorithm>

int main()
{
	std::fstream fs;

	fs.open("test", std::fstream::in);

	if(!fs)
	{
		std::cout << "failed to read in file" << std::endl;
	}
	
	int num_lines = std::count(std::istreambuf_iterator<char>(fs),
				std::istreambuf_iterator<char>(),
				'\n');

	std::cout << "Number of lines " << num_lines << std::endl;
	
	fs.close();
	fs.open("test", std::fstream::in);

	char s[256];
	
	fs.getline(s,256);

	std::stringstream ss(s);

	int per_line = std::count(std::istreambuf_iterator<char>(ss),
				std::istreambuf_iterator<char>(),
				' ') + 1;

	std::cout << "Numbers per line " << per_line << std::endl;
	
	fs.close();
	fs.open("test", std::fstream::in);

	using namespace boost::numeric::ublas;
	matrix<double> m(num_lines, per_line);

	std::cout << m << std::endl;

	int i = 0;
	while( !fs.eof() && i < num_lines )
	{
		std::cout << i << std::endl;
		int j = 0;
		char line[256];
		fs.getline(line,256);
		std::stringstream sss(line);
		while( !sss.eof() )
		{
			char num[16];
			sss.getline(num,16,' ');
			float val = atof(num);
			m(i,j) = val;
			j++;
			std::cout << j << std::endl;
		}
		i++;
	}

	std::cout << m << std::endl;

	return 0;
}
