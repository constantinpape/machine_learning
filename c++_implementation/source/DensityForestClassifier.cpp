#include <random>
#include <chrono>

#include "DensityForestClassifier.h"

using namespace boost::numeric::ublas;
	
DensityForestClassifier::DensityForestClassifier() : mTrained(false),
													 mNumInstances(0),
													 mNumClasses(0),
													 mNumTrees(15),      // numTrees default value: 15 TODO How big should this be?
													 mTrees(mNumTrees)
{}
	
void DensityForestClassifier::train(const image_data_t & train_data, const label_data_t & train_label)
{
	if( mTrained )
	{
		std::cout << "DensityForestClassifier::train: Retraining the classifier" << std::endl;
	}
// get number of instances and classes
	assert(train_data.size1() == train_label.size() );
	mNumInstances = train_data.size1();
	auto min_max	= std::minmax_element( train_label.begin(), train_label.end() );
	mNumClasses    = (*min_max.second - *min_max.first) + 1;
// instantiate and seed random generator
	unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine gen(seed);
// Clear the Forrest
	mTrees.clear();
	mTrees.resize(mNumTrees);
// Calculate the bootstrap size
	size_t bootstrap_size = mNumInstances / mNumTrees; // TODO is this reasonable ??
// instantiate uniform int distribution (0 to data size)
	std::uniform_int_distribution<size_t> distr( 0, mNumInstances );
// Train the individual trees
	for( size_t t = 0; t < mNumTrees; t++ )
	{
// make bootstrapsample for this tree
		image_data_t bootstrap_data( bootstrap_size, train_data.size2() );
		label_data_t bootstrap_label(bootstrap_size );
		for( size_t i = 0; i < bootstrap_size; i++ )
		{
			size_t indx = distr(gen);
			matrix_row<matrix<double> >  	  data_trgt( bootstrap_data, i);
			matrix_row<matrix<double> const>  data_cpy(  train_data, indx);
			std::copy( data_cpy.begin(), data_cpy.end(), data_trgt.begin() );
			bootstrap_label[i] = train_label[indx];  
		}
// train the tree on the bootstrap sample
		mTrees[t].train(bootstrap_data, bootstrap_label);
	}
	mTrained = true;
}
	
label_data_t DensityForestClassifier::predict(const image_data_t & test_data)
{
	if( !mTrained )
	{
		throw( std::runtime_error("DensityTreeForest::predict: Called before training the Classifier!") );
	}
	label_data_t labels_return;
// mNumTrees != mTrees.size() may happen, if set_number_trees is used wrongly
	if( mNumTrees != mTrees.size() )
	{
		std::cout << "Resetting mNumTrees to mTrees.size() = " << mTrees.size() << std::endl;
		mNumTrees = mTrees.size();
	}
// iterate over the trees
	std::vector<label_data_t> tree_votes;
	for( size_t t = 0; t < mNumTrees; t++)
	{
// get predictions of each tree
		tree_votes.push_back( mTrees[t].predict(test_data) );
	}
// iterate over the data, for each sample classify as majority vote
	for( size_t i = 0; i < test_data.size1(); i++ )
	{
		std::vector<short> votes;
		for(size_t t = 0; t < mNumTrees; t++)
		{
			votes.push_back( tree_votes[t][i] );
		}
		short majority_class = 0;
		size_t max_count = 0;
		for(size_t c = 0; c < mNumClasses; c++)
		{
			size_t count = std::count( votes.begin(), votes.end(), c);
			if( count > max_count )
			{
				max_count = count;
				majority_class = c;
			}
		}
		labels_return.push_back(majority_class);
	}
	return labels_return;
}

//TODO
image_data_t DensityForestClassifier::generate(const size_t N, const short label)
{
	if( !mTrained )
	{
		throw( std::runtime_error("DensityTreeForest::generate: Called before training the Classifier!") );
	}
	image_data_t data_return;

	return data_return;
}

void DensityForestClassifier::set_maximal_depth(const size_t max_depth)
{
	for( DensityTreeClassifier tree : mTrees )
	{
		tree.set_maximal_depth( max_depth );
	}
}
	
size_t DensityForestClassifier::get_maximal_depth() const
{
	return mTrees[0].get_maximal_depth();
}
	
void DensityForestClassifier::set_number_trees(const size_t num_trees)
{
	if( mTrained )
	{
		std::cout << "DensityForrestClassifier was already trained. Actual number of trees will only change when calling train again." 
			<< std::endl;
	}
	mNumTrees = num_trees;
}
	
size_t DensityForestClassifier::get_number_trees() const
{
	return mNumTrees;
}
	
void DensityForestClassifier::set_nearest_neighbors(const size_t num_nbrs)
{
	for( DensityTreeClassifier tree : mTrees )
	{
		tree.set_nearest_neighbors( num_nbrs );
	}
}
	
size_t DensityForestClassifier::get_nearest_neighbors() const
{
	return mTrees[0].get_nearest_neighbors();
}
