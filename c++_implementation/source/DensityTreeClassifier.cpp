#include <algorithm>
#include <chrono>
#include <iterator>
#include <random>
#include <fstream>
#include <string>

#include "DensityTreeClassifier.h"

using namespace boost::numeric::ublas;
	
DensityTreeClassifier::DensityTreeClassifier() : 	mTrained(false),
													mNum_instances(0),
												 	mNum_classes(0),
													mNum_dimensions(0),
													mDepth_max(10),				// 4 == default value for maximal depth of the tree
													mSplits(split_t::def),
													mRecord_splits(true),
													mDim_shuffle(false),
													mNum_shuffle(5),
													mNearest_neighbors(15), // 15 == default value for nearest neighbors in gradient approx.
													mRadius(10.),
													mTrees(),
													mPriors()
{}

void DensityTreeClassifier::train(const image_data_t & train_data, const label_data_t & train_label)
{
	if(mTrained)
	{
		std::cout << "DensityTreeClassifier::train: Retraining the classifier " << std::endl;
	}
	mTrees.clear();
	mPriors.clear();
// get number of instances, dimensions and classes
	mNum_instances	= train_data.size1();
	mNum_dimensions = train_data.size2();	
	auto min_max	= std::minmax_element( train_label.begin(), train_label.end() );
	mNum_classes    = (*min_max.second - *min_max.first) + 1;
// build the tree for each class seperately
	for( size_t c = 0; c < mNum_classes; c++)
	{
		std::cout << "DensityTreeClassfier::train: start building tree for class " << c << std::endl;
// number of instances beloniging to this class
		size_t N_class = std::count( train_label.begin(), train_label.end(), c);
// minimal number of instnaces in a node
		size_t N_min  = std::cbrt(N_class);
// calculate the prior for this class
		mPriors.push_back( N_class/static_cast<double>(mNum_instances) );
// extract the data belonging to this class
		image_data_t data_class(N_class, mNum_dimensions);
		size_t class_count = 0;
		for(size_t i = 0; i < mNum_instances; i++)
		{
// copy the data if it has the correct class
			if( train_label[i] == c )    //WARNING: comparison of int and unsigned int
			{
				matrix_row<matrix<double> const> orig_instance(train_data, i);
				matrix_row<matrix<double> >		 to_copy(data_class, class_count);
				std::copy( orig_instance.begin(), orig_instance.end(), to_copy.begin() );
				class_count++;
			}
		}
		assert(class_count == N_class);
// now build the tree for this class
// initialize the stack for the nodes
		std::vector<node_t*> stack;
// initialize the root node
		node_t* root(new node_t);
		root->set_data(data_class);
		stack.push_back(root);
		size_t count = 0;
// build the tree
		while( !stack.empty() )
		{
// pop the last node from the stack
			node_t* curr_node = *(stack.end() - 1);
			stack.pop_back();
// check whether this node is terminal
			if( terminate_depth(curr_node) || curr_node->get_data().size1() < N_min )
			{
				curr_node->calculate_probability(N_class);
				curr_node->set_terminal(true);
			}
// esle split the node, assign the children nodes and put them on the stack
			else
			{
				std::array<node_t*,2> children;
				switch( mSplits )
				{
					case def 		: children = split_node_default( curr_node, mDim_shuffle, mNum_shuffle, mRecord_splits);  
									  break;
					case def_alt 	: children = split_node_alt( curr_node, mDim_shuffle, mNum_shuffle, mRecord_splits);  
									  break;
					case gradient	: children = split_node_gradient(curr_node, mNearest_neighbors, mRecord_splits);
									  break;
					case graph		: children = split_node_graph(	 curr_node, mRadius, mRecord_splits);
									  break;
					default 		: throw std::runtime_error("Flag for split not valid");
				}
				node_t * child_left  = children[0];
				node_t * child_right = children[1];
				child_left->set_depth(curr_node->get_depth() + 1);
				child_right->set_depth(curr_node->get_depth() + 1);
				curr_node->add_child( child_left);
				curr_node->add_child( child_right);
				stack.push_back( child_left  );
				stack.push_back( child_right );
			}
			count++;
			std::cout << "DensityTreeClassifier::train: Building Tree for class " << c << " iteration "  << count << std::endl; 
		}
		mTrees.push_back(root);
	}	
	mTrained = true;
}


// Termination criterion depending on depth of node
bool DensityTreeClassifier::terminate_depth(const node_t * node)
{
// terminate the node if it is empyt
	if( node->get_data().size1() == 0 )
	{
		return true;
	}
// check the max_depth-criterion
	if( node->get_depth() >= mDepth_max )
	{
		return true;
	}
	return false;
}

label_data_t DensityTreeClassifier::predict(const image_data_t & test_data)
{
	if( !mTrained )
	{
		throw std::runtime_error("DensityTreeClassifier::predict: called before calling train!");
	}
	label_data_t label_return;
// iterate over the test data
	for(size_t i = 0; i < test_data.size1(); i++)
	{
		std::vector<double> probabilities;
// iterate over the classes to find the class with highest probability
		for(size_t c = 0; c < mNum_classes; c++)
		{
			matrix_row<matrix<double> const> data_aux(test_data,i);
			vector<double> data_point( data_aux.size() );
			std::copy(data_aux.begin(), data_aux.end(), data_point.begin() );
// find the leaf-node for this data point
			const node_t & node = search_tree(data_point,c);
			probabilities.push_back( mPriors[c] * node.get_probability() );
		}
// find the most probable class
		auto max_element = std::max_element( probabilities.begin(), probabilities.end() );
		size_t max_class = std::distance( 	 probabilities.begin(), max_element );
		label_return.push_back(max_class);
	}
	return label_return;
}

// search the tree for the leaf-node which has the data_point
node_t DensityTreeClassifier::search_tree(const vector<double> & data_point, const size_t c )
{
// get the root node of the tree belonging to this class
	node_t * curr_node = mTrees[c];
// walk the tree until we come to a terminal node	
	while( !curr_node->get_terminal() )
	{
		size_t dim 		= curr_node->get_split_dimension();
		double thresh 	= curr_node->get_split_threshold();
// look whether this data_point is left or right of the split boundary
		if( data_point(dim) < thresh )
		{
			curr_node = curr_node->get_child( node_t::side_t::left );
		}
		else
		{
			curr_node = curr_node->get_child( node_t::side_t::right );
		}
	}
	return *curr_node;
}

// generate N instances of class given by label
image_data_t DensityTreeClassifier::generate(const size_t N, const short label)
{
	if( !mTrained )
	{
		throw std::runtime_error("DensityTreeClassifier::generate: called before calling train!");
	}
	image_data_t data_return( N, mNum_dimensions );
// instantiate and seed random generator
	unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine gen(seed);
// instantiate uniform real distribution (0 to 1)
	std::uniform_real_distribution<double> distr(0.0,1.0);
// generate N new instances
	for( size_t i = 0; i < N; ++i)
	{
// get the root node of the class to generate
		node_t * curr_node = mTrees[label];
// walk the tree until we get to a leaf-node
		while( !curr_node->get_terminal() )
		{
			size_t N = curr_node->get_data().size1();
			node_t * l_node = curr_node->get_child( node_t::side_t::left );
			node_t * r_node = curr_node->get_child( node_t::side_t::right);
// calculate p_left
			size_t N_l = l_node->get_data().size1();
			double V_l = l_node->get_volume();
			double p_l = N_l / (N * V_l);
// calculate p_right
			size_t N_r = r_node->get_data().size1();
			double V_r = r_node->get_volume();
			double p_r = N_r / (N * V_r);
// calculate p and q (normalised probabilities)
			double p = p_l / (p_l + p_r);
// go left with prob p, right with prob q
			if( p < distr(gen) )
			{
				curr_node = l_node;
			}
			else
			{
				curr_node = r_node;
			}
		}
// sample uniformly from the leaf-node we ended up in 
		const image_data_t & sample_data = curr_node->get_data();
		for( size_t d = 0; d < mNum_dimensions; d++)
		{
			matrix_column<matrix<double> const> data_dim( sample_data, d );
			double max_val = *( std::max_element( data_dim.begin(), data_dim.end() ) );
			double min_val = *( std::min_element( data_dim.begin(), data_dim.end() ) );
			std::uniform_real_distribution<double> distr_dim( min_val, max_val );
			data_return(i,d) = distr_dim(gen);
		}
	}
	return data_return;
}
	
double DensityTreeClassifier::get_likelihood(const vector<double> & data, const short label)
{
	if( !mTrained )
	{
		throw std::runtime_error("DensityTreeClassifier::get_likelihood: called before calling train!");
	}
	const node_t & node = search_tree(data,label);
	return node.get_probability();
}
	
void DensityTreeClassifier::set_split(const split_t split)
{
	mSplits = split;
}
	
DensityTreeClassifier::split_t DensityTreeClassifier::get_split() const
{
	return mSplits;
}
	
void DensityTreeClassifier::set_maximal_depth(const size_t max_depth)
{
	mDepth_max = max_depth;
}

size_t DensityTreeClassifier::get_maximal_depth() const
{
	return mDepth_max;
}
	
void DensityTreeClassifier::set_nearest_neighbors(const size_t num_nbrs)
{
	mNearest_neighbors = num_nbrs;
}
	
size_t DensityTreeClassifier::get_nearest_neighbors() const
{
	return mNearest_neighbors;
}
	
void DensityTreeClassifier::set_shuffle(const bool enable, const size_t num_shuffle)
{
	mDim_shuffle = enable;
	mNum_shuffle = num_shuffle;
}

void DensityTreeClassifier::set_record_split(const bool enable)
{
	mRecord_splits = enable;
}

bool DensityTreeClassifier::get_record_split() const
{
	return mRecord_splits;
}
	
void DensityTreeClassifier::set_radius(const double radius)
{
	mRadius = radius;
}

double DensityTreeClassifier::get_radius() const
{
	return mRadius;
}
