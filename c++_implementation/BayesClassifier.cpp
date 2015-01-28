#include <algorithm>
#include <exception>
#include <iostream>
#include <iterator>
#include <math>

#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>

#include "BayesClassifier.h"

using namespace boost::numeric::ublas;

BayesClassifier::BayesClassifier()
{
	trained 		= false;
	num_instances   = 0;
	num_dimensions 	= 0;
	num_classes 	= 0;
}

void BayesClassifier::train(const image_data_t & train_data, const label_data_t & train_label)
{
	if( trained )
	{
		std::cout << "Called BayesClassifier::train with trained = true, retraining the classifier." << std::endl;
	}
// determine number of instances, dimensions and classes
	num_instances	= train_data.size1();
	num_dimensions 	= train_data.size2(); 
// get number of classes, here we assume that class labeling starts at zero and there are no intermediate numbers missing !!!
	min_max			= std::minmax(train_label)
	num_classes    	= (min_max.second - min_max.first) + 1
// calculate the number of class instances and the priors
	instances_per_class.clear();
	priors.clear();
	for( size_t i = 0; i < num_classes; i++)
	{
		size_t N_class = std::count( train_label.begin(), train_label.end(), i);
	  	instances_per_class.push_back(N_class);
		priors.push_back( N_class/static_cast<double>(num_instances) );
	}
// get the optimal bin width for each dimensions
	bins.clear();
	irrelevant_dims.clear();
	for( size_t d = 0; d < num_dimensions; d++ )
	{
		bin_t curr_bin = get_optimal_bins(train_data,d);
		if( curr_bin.width == 0.0)
		{
			irrelevant_dims.push_back(d); 
		}
		bins.push_back(current_bin);
	}
// initialize the histograms
	histograms.clear() 
	for( size_t d = 0; d < num_dimensions; d++)
	{
		histo_dim = histogram_t(bins[d].num_bins, num_classes);
		histograms.push_back(histo_dim);
	}
// calculate the histograms from the data
	for( size_t d = 0; d < num_dimensions; d++)
	{
// check if this dimension is irrelevant and continue if it is
		if( std::find( irrelevant_dims.begin(), irrelevant_dims.end(), d) == irrelevant_dims.end() )
		{
			continue;
		}
		else
		{
// iterate over the training data and assign to corresponding bin
			for( size_t i = 0; i < num_instances; i++)
			{
				double val 		= train_data(i,d);
				uint8_t label 	= train_label[i];
				// find the bin
				size_t bin = static_cast<size_t>( ( (val - bins[d].lowest_val) / bins[d].val_range) * bins[d].num_bins );
				histograms[d](bin,label) += 1;
			}	
// normalise the histogram
			for( size_t c = 0; c < num_classes, c++)
			{
				row( histograms[d](c) ) /= instances_per_class[c]; // TODO also divide by bin width ???
			}
		}
	}
// set train flag to true
	trained = true;
}

const label_data_t & BayesClassifier::predict(const image_data_t & test_data)
{
	label_data_t data_return;
	if( !trained )
	{
		throw std::runtime_error("BayesClassifier::predict trying to predict without having trained!");
	}
// iterate over test instances
	for( size_t i = 0; i < test_data.size1(); ++i)
	{
// calculate the likelihood for each class
		std::array<double,num_classes> class_likelihoods;
		for( size_t c = 0; c < num_classes; ++c)
		{
			double likelihood = 1.0;
			for( size_t d = 0; d < num_dimensions; d++)
			{
				double val = test_data(i,d);
// check if this dimension is irrelevant and continue if it is
				if( std::find( irrelevant_dims.begin(), irrelevant_dims.end(), d) == irrelevant_dims.end() )
				{
					continue;
				}
				else
				{
					size_t bin = static_cast<size_t>( ( (val - bins[d].lowest_val) / bins[d].val_range) * bins[d].num_bins );
					likelihood *=  histograms[d](bin,c);
				}
			}
			class_likelihoods[c] = likeliehood;
		}
// find the class with biggest likelihood
		auto max = std::max_element(class_likelihoods.begin(), class_likelihoods.end());
		size_t max_class = std::distance(class_likelihoods.begin(), max);
		data_return.push_back(max_class);
	}
	return data_return;
}

void BayesClassifier::compute_cdf()
{

}

// TODO implement different bin width if issues occur
bin_t BayesClassifier::get_optimal_bins(const image_data_t & train_data, size_t dim)
{
	bin_t bin;
// calculate the optimal bin width with IRQ-criterion
// first get the correct dimension == column
	matrix_column<double> data_dim_aux(train_data,d);
// copy all data to a vector to safely manipulate it
	vector<double> data_dim( data_dim_aux.size() );
	std::copy(data_dim_aux.begin(), data_dim_aux.end(), data_dim.begin() );
// sort the data in this dimension
	std::sort(data_dim.begin(), data_dim.end())
// get lower quartile
	size_t size    	= num_instances;
	size_t ind_1   	= size / 4;
	double q_1     	= (size%2 == 0) ? ( data_dim(ind_1+1) + data_dim(ind_1) )/2 : data_dim(ind_1);
// get upper quartile
	size_t ind_3   	= 3*size / 4;
	double q_3     	= (size%2 == 0) ? ( data_dim(ind_3+1) + data_dim(ind_3) )/2 : data_dim(ind_3+1);
// calculate the iqr and bin width
	double iqr	   	= q_3 - q_1
	bin.width 		= 2*iqr / std::cbrt(num_instances);
// get lowest and highest value in the dimension
	bin.lowest_val  = *( data_dim.begin() );
	bin.highest_val = *( data_dim.end() - 1 );
// get number of bins 
	bin.val_range   = bin.highest_val - bin.lowest_val;
	bin.num_bins 	= static_cast<size_t>(bin.val_range / bin.width);
//limit number of bins
	if(bin.num_bins > std::sqrt(num_instances) )
	{
		std::cout << "Number of bins " << bin.num_bins << " bigger than sqrt(N) = " << std::sqrt(num_instances) << std::endl;
		// TODO change number of bins
	}
	return bin;
}
