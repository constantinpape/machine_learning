#include <iostream>
#include <fstream>
#include <exception>
#include <boost/numeric/ublas/io.hpp>

#include "common.h"
	
using namespace boost::numeric::ublas;

image_data_t read_mnist_data(std::string fname)
{
// open filestream to read in the data
	std::fstream fs;
	fs.open(fname, std::fstream::in);
	if(!fs)
	{
		throw std::runtime_error("Filestream not openend correctly!");
	}
// determine number of lines and number of values per line
	int num_lines = std::count(std::istreambuf_iterator<char>(fs),
				std::istreambuf_iterator<char>(),
				'\n');
	std::cout << "Reading " << fname << ": Number of instances: " << num_lines << std::endl;
	fs.close();
	fs.open(fname, std::fstream::in);
	char s[1024];
	fs.getline(s,1024);
	std::stringstream ss(s);
	int per_line = std::count(std::istreambuf_iterator<char>(ss),
				std::istreambuf_iterator<char>(),
				' ') + 1;
	std::cout << "Reading " << fname <<": Number of dimensions: " << per_line << std::endl;
	fs.close();
	fs.open(fname, std::fstream::in);
// initialise the data
	 image_data_t data_return(num_lines, per_line);
// read in the data
	int i = 0;
	while( !fs.eof() && i < num_lines )
	{
		int j = 0;
		char line[1024];
		fs.getline(line,1024);
		std::stringstream sss(line);
		while( !sss.eof() )
		{
			char num[16];
			sss.getline(num,16,' ');
			double val = atof(num);
			data_return(i,j) = val;
			j++;
		}
		i++;
	}
	return data_return;
}

label_data_t read_mnist_label(std::string fname)
{
// open filestream to read in the data
	std::fstream fs;
	fs.open(fname, std::fstream::in);
	if(!fs)
	{
		throw std::runtime_error("Filestream not openend correctly!");
	}
// determine number of lines
	int num_lines = std::count(std::istreambuf_iterator<char>(fs),
				std::istreambuf_iterator<char>(),
				'\n');
	std::cout << "Reading " << fname << ": Number of instances: " << num_lines << std::endl;
	fs.close();
	fs.open(fname, std::fstream::in);
// init return data
	label_data_t label_return;
// read in the data
	int i = 0;
	while( !fs.eof() && i < num_lines )
	{
		char num[16];
		fs.getline(num,16);
		uint8_t val = atoi(num);
		label_return.push_back(val);
		i++;
	}
	return label_return;
}
