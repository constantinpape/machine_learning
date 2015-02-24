#include <algorithm>
#include <chrono>
#include <exception>
#include <iostream>
#include <iterator>
#include <math.h>
#include <random>

#include "BayesClassifier.h"

using namespace boost::numeric::ublas;

BayesClassifier::BayesClassifier() : 	trained(false),
   										num_instances(0),
										num_dimensions(0),
										num_classes(0),
										instances_per_class(),
										priors(),
										bins(),
										irrelevant_dims(),
										histograms(),
										cdfs()
{}

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
	auto min_max	= std::minmax_element( train_label.begin(), train_label.end() );
	num_classes    	= (*min_max.second - *min_max.first) + 1;
// calculate the number of class instances and the priors
	instances_per_class.clear();
	priors.clear();
	for( size_t c = 0; c < num_classes; c++)
	{
		size_t N_class = std::count( train_label.begin(), train_label.end(), c);
	  	instances_per_class.push_back(N_class);
		priors.push_back( N_class/static_cast<double>(num_instances) );
	}
// get the optimal bin width for each dimensions
	bins.clear();
	irrelevant_dims.clear();
	for( size_t d = 0; d < num_dimensions; d++ )
	{
		bin_t current_bin = get_optimal_bins(train_data,d);
// debug output
                //std::cout << "dimension: " << d << "\twidth: " << current_bin.width << "\tnumber: " << current_bin.num_bins << std::endl;
		if( current_bin.width == 0.0)
		{
			irrelevant_dims.push_back(d); 
		}
		bins.push_back(current_bin);
	}
// initialize the histograms
	histograms.clear(); 
	for( size_t d = 0; d < num_dimensions; d++)
	{
		histogram_t histo_dim = histogram_t(bins[d].num_bins, num_classes);
		histograms.push_back(histo_dim);
	}
// calculate the histograms from the data
	for( size_t d = 0; d < num_dimensions; d++)
	{
// check if this dimension is irrelevant and continue if it is
		if( std::find( irrelevant_dims.begin(), irrelevant_dims.end(), d) != irrelevant_dims.end() )
		{
			continue;
		}
		else
		{
// iterate over the training data and assign to corresponding bin
			for( size_t i = 0; i < num_instances; i++)
			{
				double val 	= train_data(i,d);
				short label = train_label[i];
				// find the bin
				size_t bin = static_cast<size_t>( ( (val - bins[d].lowest_val) / bins[d].val_range) * bins[d].num_bins );
// lower bin by 1 if it is too big (this may happen...) FIXME
				if( bin == bins[d].num_bins)
				{
					bin--;
				}
// debug output
				//std::cout << "Dim " << d << " instance " << i << " bin " << bin << " bin-max " << bins[d].num_bins << std::endl;
				histograms[d](bin,label) += 1;
			}	
// normalise the histogram
			for( size_t c = 0; c < num_classes; c++)
			{
				row( histograms[d], c ) /= instances_per_class[c]; // TODO also divide by bin width ???
			}
		}
	}
// create the cumulative densities
	compute_cdf();
// set train flag to true
	trained = true;
}

label_data_t BayesClassifier::predict(const image_data_t & test_data)
{
	if( !trained )
	{
		throw std::runtime_error("BayesClassifier::predict trying to predict without having trained!");
	}
	label_data_t labels_return;
// iterate over test instances
	for( size_t i = 0; i < test_data.size1(); ++i)
	{
// calculate the likelihood for each class
		std::vector<double> class_likelihoods;
		for( size_t c = 0; c < num_classes; ++c)
		{
			double likelihood = 1.0;
			for( size_t d = 0; d < num_dimensions; d++)
			{
				double val = test_data(i,d);
// check if this dimension is irrelevant and continue if it is
				if( std::find( irrelevant_dims.begin(), irrelevant_dims.end(), d) != irrelevant_dims.end() )
				{
					continue;
				}
				else
				{
					size_t bin = static_cast<size_t>( ( (val - bins[d].lowest_val) / bins[d].val_range) * bins[d].num_bins );
// set bin to binmax if it is bigger than binmax, this may happen because we havent seen the test_data yet
					if( bin >= bins[d].num_bins)
					{
						bin = bins[d].num_bins - 1;
					}
					likelihood *=  histograms[d](bin,c);
				}
			}
// multiply with the prior
			likelihood *= priors[c];
			class_likelihoods.push_back(likelihood);
		}
// find the class with biggest likelihood
		auto max = std::max_element(class_likelihoods.begin(), class_likelihoods.end());
		size_t max_class = std::distance(class_likelihoods.begin(), max);
		labels_return.push_back(max_class);
	}
	return labels_return;
}

image_data_t BayesClassifier::generate(const size_t N, const short label)
{
	if( !trained )
	{
		throw std::runtime_error("BayesClassifier::generate: Trying to generate without having trained!");
	}
	image_data_t data_return(N,num_dimensions);
// instantiate and seed random generator
	unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine gen(seed);
// instantiate uniform real distribution (0 to 1)
	std::uniform_real_distribution<double> distr(0.0,1.0);
// generate N new instances
	for( size_t i = 0; i < N; ++i)
	{
		for( size_t d = 0; d < num_dimensions; d++ )
		{
// check if this dimension is irrelevant and set data to zero if it is
			if( std::find( irrelevant_dims.begin(), irrelevant_dims.end(), d) != irrelevant_dims.end() )
			{
				data_return(i,d) = 0.;
			}
// sampling strategy: sample from the 2 most likely histogram bins with their prob.
// then sample uniform inside the chosen bin
			else
			{
// get data and copy it to safely operate on it
				matrix_column<matrix<double> const> histo(histograms[d], label);
				assert(histo.size() == bins[d].num_bins);
				vector<double> histo_sort( histo.size() );
				std::copy( histo.begin(), histo.end(), histo_sort.begin() );
// sort bins by their probability
				std::sort( histo_sort.begin(), histo_sort.end() );
// get probabilities of the 2 bins with highest probability
				double p = *(histo_sort.end() - 1);
				double q = *(histo_sort.end() - 2);
// normalise the probability
				double p_1 = p / (p + q);
// decide which bin according to the probs
				size_t bin_chosen = 0;
				if( p_1 > distr(gen) )
				{
					auto search_bin = std::find(histo.begin(), histo.end(), p);
					bin_chosen 		= std::distance(histo.begin(), search_bin);
				}
				else
				{
					auto search_bin = std::find(histo.begin(), histo.end(), q);
					bin_chosen 		= std::distance(histo.begin(), search_bin);
				}
// sample uniform inside the bin
				double bin_min = bin_chosen * bins[d].width + bins[d].lowest_val;	
				double bin_max = (bin_chosen + 1) * bins[d].width + bins[d].lowest_val;	
				std::uniform_real_distribution<double> distr_bin(bin_min, bin_max);
				data_return(i,d) = distr_bin(gen);
			}
		}	
	}
	return data_return;
}

vector<double> BayesClassifier::inverse_cdf(const matrix_row<matrix<double> > & data, const short label)
{
	if( !trained )
	{
		throw std::runtime_error("BayesClassifier::inverse_cdf: Trying to calculate the inverse cdf without having trained!");
	}	
	assert(data.size() == num_dimensions);
	vector<double> data_return(num_dimensions);
// iterate over the dimensions
	for( size_t d = 0; d < num_dimensions; d++)
	{
// check if this dimension is irrelevant and set data to zero if it is
		if( std::find( irrelevant_dims.begin(), irrelevant_dims.end(), d) != irrelevant_dims.end() )
		{
			data_return(d) = 0.;
		}
		else
		{
// get the CDF of this dimension and class
			const std::vector<double> & cdf_dim = cdfs[d][label]; 
			double val_dim = data(d);
// find the bin that val_dim is in	
			size_t b = 0;
			while( b < cdf_dim.size() )
			{
				if( val_dim < cdf_dim[b] )
				{
					break;
				}
				b++;
			}
// the value of the inverse cdf corresponds to the position of the middle of the bin
// TODO + 0.5 or - 0.5 ??? I think it is correct this way, but not quite sure
			data_return(d) = bins[d].width * (b - 0.5);
		}
	}
	return data_return;
}
	
double BayesClassifier::get_likelihood(const matrix_row<matrix<double> const> & data, const short label)
{
	if( !trained )
	{
		throw std::runtime_error("BayesClassifier::get_likelihood: called before calling train!");
	}	
	double likelihood = 1.0;
	for( size_t d = 0; d < num_dimensions; d++)
	{
		double val = data(d);
// check if this dimension is irrelevant and continue if it is
		if( std::find( irrelevant_dims.begin(), irrelevant_dims.end(), d) != irrelevant_dims.end() )
		{
			continue;
		}
		else
		{
			size_t bin = static_cast<size_t>( ( (val - bins[d].lowest_val) / bins[d].val_range) * bins[d].num_bins );
// set bin to binmax if it is bigger than binmax, this may happen because we havent seen the test_data yet
			if( bin >= bins[d].num_bins)
			{
				bin = bins[d].num_bins - 1;
			}
			likelihood *=  histograms[d](bin,label);
		}
	}
	return likelihood;
}
	
vector<double> BayesClassifier::get_cdf(const matrix_row<matrix<double> const> & data, const short label)
{
	if( !trained )
	{
		throw std::runtime_error("BayesClassifier::get_cdf: called before calling train!");
	}	
	assert(data.size() == num_dimensions);
	vector<double> cdf_return(num_dimensions);
	for( size_t d = 0; d < num_dimensions; d++)
	{
		if( std::find( irrelevant_dims.begin(), irrelevant_dims.end(), d) != irrelevant_dims.end() )
		{
			cdf_return(d) = 0.;
		}
		else
		{
			double val = data(d);
			size_t bin = static_cast<size_t>( ( (val - bins[d].lowest_val) / bins[d].val_range) * bins[d].num_bins );
// set bin to binmax if it is bigger than binmax, this may happen because we havent seen the test_data yet
			if( bin >= bins[d].num_bins)
			{
				bin = bins[d].num_bins - 1;
			}
			cdf_return(d) = cdfs[d][label][bin];	
		}
	}
	return cdf_return;
}

void BayesClassifier::compute_cdf()
{
	cdfs.clear();
	for(size_t d = 0; d < num_dimensions; d++)
	{
		cdf_t cdf_d;	
		if( std::find( irrelevant_dims.begin(), irrelevant_dims.end(), d) != irrelevant_dims.end() )
		{
			for(size_t c = 0; c < num_classes; c++)
			{
				cdf_d.push_back( std::vector<double>{{0}} );
			}
		}
		else
		{
			const auto & histo 	= histograms[d];
		   	size_t n_bins 		= bins[d].num_bins;
			assert(histo.size1() == n_bins);
			for(size_t c = 0; c < num_classes; c++)
			{
				std::vector<double> cdf_c;
				cdf_c.push_back( histo(0,c) );
				for(size_t b = 1; b < n_bins; b++)
				{
					double val = cdf_c[b-1] + histo(b,c);
					cdf_c.push_back(val);	
				}	
				cdf_d.push_back(cdf_c);
			}
		}
		cdfs.push_back(cdf_d);
	}
}

// TODO what to do for too small bins <-> too many bins
bin_t BayesClassifier::get_optimal_bins(const image_data_t & train_data, const size_t dim)
{
	bin_t bin;
// calculate the optimal bin width with IRQ-criterion
// first get the correct dimension == column
	matrix_column<matrix<double> const> data_dim_aux(train_data, dim);
// copy all data to a vector to safely manipulate it
	vector<double> data_dim( data_dim_aux.size() );
	std::copy(data_dim_aux.begin(), data_dim_aux.end(), data_dim.begin() );
// sort the data in this dimension
	std::sort(data_dim.begin(), data_dim.end());
// get lower quartile
	size_t size    	= num_instances;
	size_t ind_1   	= size / 4;
	double q_1     	= (size%2 == 0) ? ( data_dim(ind_1+1) + data_dim(ind_1) )/2 : data_dim(ind_1);
// get upper quartile
	size_t ind_3   	= 3*size / 4;
	double q_3     	= (size%2 == 0) ? ( data_dim(ind_3+1) + data_dim(ind_3) )/2 : data_dim(ind_3+1);
// calculate the iqr and bin width
	double iqr	   	= q_3 - q_1;
	bin.width 		= 2.*iqr / std::cbrt(num_instances);
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
		std::cout << "IQR " << iqr << " N^1/3 " << std::cbrt(num_instances) << std::endl; 
		bin.num_bins = static_cast<size_t>( std::sqrt(num_instances) );
		bin.width    = bin.val_range / bin.num_bins;
		std::cout << "Changing number of bins to: " << bin.num_bins << std::endl;
	}
	return bin;
}
