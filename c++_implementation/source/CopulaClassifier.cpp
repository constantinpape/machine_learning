#include "CopulaClassifier.h"

using namespace boost::numeric::ublas;
	
CopulaClassifier::CopulaClassifier() : mTrained(false),
									   mNum_instances(0),
									   mNum_classes(0),
									   mNum_dimensions(0),
									   mPriors(),
									   mBayes(),
									   mDensityTree()
{}

void CopulaClassifier::train(const image_data_t & train_data, const label_data_t & train_label)
{
	if(mTrained)
	{
		std::cout << "CopulaClassifier::train: retraining the classifier!" << std::endl;	
	}
	mNum_instances  = train_data.size1();
	auto min_max	= std::minmax_element( train_label.begin(), train_label.end() );
	mNum_classes    = (*min_max.second - *min_max.first) + 1;
	mNum_dimensions = train_data.size2();
// calculate the priors
	for( size_t c = 0; c < mNum_classes; c++)
	{
		size_t N_class = std::count( train_label.begin(), train_label.end(), c);
		mPriors.push_back( N_class/static_cast<double>(mNum_instances) );
	}
// train the BayesClassifier on the original data
	mBayes.train(train_data, train_label);
// calculate the copula of the data
	image_data_t data_copula = get_copula(train_data);
// train the DensityTree on the copula data
	mDensityTree.train(data_copula, train_label);
	mTrained = true;
}

// get the copula of data
image_data_t CopulaClassifier::get_copula(const image_data_t & data)
{
	image_data_t data_copula( data.size1(), data.size2() );
// iterate ove the dimensions
	for( size_t d = 0; d < data.size2(); d++)
	{
		matrix_column<matrix<double> const> data_dim( data, d );
// get the indices that would sort the data in this dimension
		vector<double> arguments_sorted = get_sorted_indices(data_dim);
// sort these to get the ranked indiced
		vector<double> arguments_ranked = get_sorted_indices(arguments_sorted);
// add one to the sorted data
		vector<double> ones( data.size1() );
		std::fill(ones.begin(), ones.end(), 1.);
		arguments_ranked += ones;
// divide by N + 1
		arguments_ranked /= ( data.size1() + 1);
// copy to the return data
		matrix_column<matrix<double> > cpy_trgt( data_copula, d );
		std::copy(arguments_ranked.begin(), arguments_ranked.end(), cpy_trgt.begin() );
	}
	return data_copula;
}

label_data_t CopulaClassifier::predict(const image_data_t & test_data)
{
	if(!mTrained)
	{
		throw("CopulaClassifier::predict: Trying to predict without having trained the classifier!");
	}
	label_data_t label_return;
// iterate over the test data
	for( size_t i = 0; i < test_data.size1(); i++ )
	{
		std::vector<double> probabilities;
		matrix_row<matrix<double> const> data_instance( test_data, i );	
// iterate over the classes
		for( size_t c = 0; c < mNum_classes; c++ )
		{
			double likelihood = 1.;
// get the likelihood for the data from the bayes classifier
			likelihood *= mBayes.get_likelihood( data_instance, c ); 
// calculate the cdf of the data
			vector<double> data_cdf = mBayes.get_cdf( data_instance, c );
// get the likelihood for the cdf_data from the density tree classifier
			likelihood *= mDensityTree.get_likelihood( data_cdf, c );
// multiply with the prior
			likelihood *= mPriors[c];
			probabilities.push_back(likelihood);
		}
// find class with highest probability
		auto max_elem 	 = std::max_element( probabilities.begin(), probabilities.end() );
		short max_class = std::distance( probabilities.begin(), max_elem);
		label_return.push_back(max_class);
	}
	return label_return;
}

image_data_t CopulaClassifier::generate(const size_t N, const short label)
{
	if(!mTrained)
	{
		throw("CopulaClassifier::generate: Trying to generate new data without having trained the classifier!");
	}
	image_data_t data_return(N, mNum_dimensions);
// generate N instances of class given by label
	image_data_t copula_generated = mDensityTree.generate(N,label);
	assert( N == copula_generated.size1() );
// convert the copula data to the original values with the inveres cdf of the Bayes classfier
	for(size_t i = 0; i < N; i++ )
	{
		const matrix_row<matrix<double> > data_instance( copula_generated, i );
		vector<double> data_orig = mBayes.inverse_cdf( data_instance, label );
		matrix_row<matrix<double> > cpy_trgt( data_return, i );
		std::copy( data_orig.begin(), data_orig.end(), cpy_trgt.begin() );
	}
	return data_return;
}

void CopulaClassifier::set_split(const DensityTreeClassifier::split_t split)
{
	mDensityTree.set_split(split);
}
	
DensityTreeClassifier::split_t CopulaClassifier::get_split() const
{
	return mDensityTree.get_split();
}

void CopulaClassifier::set_maximal_depth(const size_t max_depth)
{
	mDensityTree.set_maximal_depth(max_depth);
}

size_t CopulaClassifier::get_maximal_depth() const
{
	return mDensityTree.get_maximal_depth();
}
	
void CopulaClassifier::set_nearest_neighbors(const size_t num_nbrs)
{
	mDensityTree.set_nearest_neighbors(num_nbrs);
}
	
size_t CopulaClassifier::get_nearest_neighbors() const
{
	return mDensityTree.get_nearest_neighbors();
}
	
void CopulaClassifier::set_record_split(const bool enable)
{
	mDensityTree.set_record_split(enable);
}

bool CopulaClassifier::get_record_split()
{
	return mDensityTree.get_record_split();
}

