#pragma once

#include <vector>
#include <boost/numeric/ublas/matrix.hpp>
#include <string>
#include <cstdint>

//typedefs for the datastructures
//TODO proper types
typedef boost::numeric::ublas::matrix<double>  image_data_t;
typedef std::vector<uint8_t> 					label_data_t;
typedef boost::numeric::ublas::matrix<size_t>  histogram_t;

// stores the information for bins in given dimenstion
struct bin_t
{
	double width;
	double lowest_val;
	double highest_val;
	double val_range;
	size_t num_bins;
};

image_data_t read_mnist_data(std::string fname);
label_data_t read_mnist_label(std::string fname);
