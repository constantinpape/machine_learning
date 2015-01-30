#pragma once

#include <vector>
#include <boost/numeric/ublas/matrix.hpp>
#include <string>
#include <cstdint>

//typedefs for the datastructures
typedef boost::numeric::ublas::matrix<double>  	image_data_t;
typedef std::vector<short> 						label_data_t;
typedef boost::numeric::ublas::matrix<double>  	histogram_t;
typedef std::vector<std::vector<double> >		cdf_t;

// stores the information for bins in given dimenstion
struct bin_t
{
	double width;
	double lowest_val;
	double highest_val;
	double val_range;
	size_t num_bins;
};

image_data_t read_mnist_data(const std::string & fname);
label_data_t read_mnist_label(const std::string & fname);

void save_data( const std::string & fname, const image_data_t & data);
void save_label(const std::string & fname, const label_data_t & label);
