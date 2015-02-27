#pragma once

#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/matrix.hpp>

//typedefs for the datastructures
typedef boost::numeric::ublas::matrix<double>  	image_data_t;
typedef std::vector<unsigned short> 			label_data_t;
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

// format output for ofstream etc.
class formatted_output
{
public:
  formatted_output(ostream & obj, int w);

  template<typename T>
  formatted_output& operator<<(const T& output);

  formatted_output& operator<<(ostream& (*func)(ostream&));

private:
  int width;
  ostream& stream_obj;
};

typedef std::pair<double,size_t> pair_t;

boost::numeric::ublas::vector<double> get_sorted_indices(
		const boost::numeric::ublas::matrix_column<boost::numeric::ublas::matrix<double> const> & data );

boost::numeric::ublas::vector<double> get_sorted_indices(
		const boost::numeric::ublas::vector<double>& data );

boost::numeric::ublas::vector<double> get_ranked_indices(
		const boost::numeric::ublas::matrix_column<boost::numeric::ublas::matrix<double> const> & data );

boost::numeric::ublas::vector<double> get_ranked_indices(
		const boost::numeric::ublas::vector<double>& data );

image_data_t read_mnist_data(const std::string & fname, size_t buff_size = 1024);
label_data_t read_mnist_label(const std::string & fname);

void save_data( const std::string & fname, const image_data_t & data);
void save_label(const std::string & fname, const label_data_t & label);
