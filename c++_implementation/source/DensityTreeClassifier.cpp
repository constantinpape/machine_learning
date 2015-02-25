#include <algorithm>
#include <chrono>
#include <iterator>
#include <random>
#include <fstream>
#include <string>

#include "DensityTreeClassifier.h"
#include "utility/splits.h"

using namespace boost::numeric::ublas;
	
DensityTreeClassifier::DensityTreeClassifier() : 	mTrained(false),
													mNum_instances(0),
												 	mNum_classes(0),
													mNum_dimensions(0),
													mDepth_max(4),				// 4 == default value for maximal depth of the tree
													mSplits(split_t::def),
													mRecord_splits(false),
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

// split node and return the two children nodes
std::array<node_t*, 2> DensityTreeClassifier::split_node(node_t * node, const size_t N_min)
{
// epsilon for the thresholds
	double eps 	= 0.01;
// get number of instances in this node
	size_t N_node = node->get_data().size1();
	double best_thresh = 0.;
	double best_gain   = 0.; 
	size_t best_dim	   = 0;
	std::vector<size_t> dimensions;
	for( size_t d = 0; d < mNum_dimensions; d++)
	{
		dimensions.push_back(d);
	}
// if mDim_shuffle == true shuffle the dimensions and only iterate over the first mNum_shuffle entries
	if( mDim_shuffle )
	{
		std::random_device rd;
		std::mt19937 g( rd() );
		std::shuffle( dimensions.begin(), dimensions.end(), g );
		dimensions.resize(mNum_shuffle);
	}
// iterate all dimensions to find the best possible split
	for( size_t d : dimensions )
	{
// sort the data in this dimension
		matrix_column<matrix<double> const> data_aux(node->get_data(), d);
		vector<double> data_dim( data_aux.size() );
		std::copy( data_aux.begin(), data_aux.end(), data_dim.begin() );
		std::sort( data_dim.begin(), data_dim.end() );
		assert( data_dim.size() == N_node );
// precompute the volume in this dimension
		double V_dim = node->get_volume();
// divide by the volume of this dimension
		double min_dim = *( data_dim.begin() );
		double max_dim = *( data_dim.end() - 1 );
		V_dim /= (max_dim - min_dim);
// calculate all threshold
		std::vector<double> thresholds;
		double min_thresh = *data_dim.begin() + eps;
		double max_thresh = *(data_dim.end()-1) - eps;
// almost superfluous since data_dim 
		for( size_t i = 1; i < N_node; i++ )
		{
			if( data_dim[i] - eps > min_thresh )
			{
				thresholds.push_back(data_dim[i] - eps);
			}	 
		}
		for( size_t i = 0; i < N_node; i++ )
		{
			if( data_dim[i] + eps < max_thresh )
			{
				thresholds.push_back(data_dim[i] + eps);
			}	 
		}
// iterate over the thresholds
		for( double t : thresholds)
		{
			auto split_iter = std::lower_bound(data_dim.begin(), data_dim.end(), t);
			size_t N_l 		= std::distance(data_dim.begin(), split_iter);
			size_t N_r 		= N_node - N_l;
// we don't want too small splits
			if( N_l < N_min || N_r < N_min)
			{
				continue;
			}
			else
			{
// calculate volumes
			double V_l = V_dim * ( t - min_dim );
			double V_r = V_dim * ( max_dim - t );
			double gain = std::pow( (static_cast<double>(N_l) / N_node), 2 ) / V_l + std::pow( (static_cast<double>(N_r) / N_node), 2 ) / V_r; 
// check whether this is the best gain so far
				if( gain > best_gain )
				{
					best_gain 	= gain;
					best_thresh = t;
					best_dim 	= d; 
				}
			}
		}
	}
// store dimension and threshold in the node	
	node->set_split_dimension(best_dim);
	node->set_split_threshold(best_thresh);
// split the data accordingly	
	matrix_column<matrix<double> const> data_dim( node->get_data(), best_dim );
	vector<double> data_sort( data_dim.size() );
	std::copy( data_dim.begin(), data_dim.end(), data_sort.begin() );
	std::sort( data_sort.begin(), data_sort.end() );
	auto split_iter = std::lower_bound(data_sort.begin(), data_sort.end(), best_thresh);
	size_t N_l 		= std::distance(data_sort.begin(), split_iter);
	size_t N_r		= N_node - N_l;
	image_data_t data_l( N_l, mNum_dimensions);
	image_data_t data_r( N_r, mNum_dimensions);
	size_t count_l = 0;
	size_t count_r = 0;
	for( size_t i = 0; i < N_node; i++)
	{
		if( node->get_data()(i,best_dim) < best_thresh ) // datapoint is on the left
		{
			matrix_row<matrix<double> const> 	copy_source( node->get_data(), i );
			matrix_row<matrix<double> >			copy_target( data_l, count_l );
			std::copy( copy_source.begin(), copy_source.end(), copy_target.begin() );
			count_l++;
		}
		else	// datapoint is on the right
		{
			matrix_row<matrix<double> const> 	copy_source( node->get_data(), i );
			matrix_row<matrix<double> >			copy_target( data_r, count_r );
			std::copy( copy_source.begin(), copy_source.end(), copy_target.begin() );
			count_r++;
		}
	}
	std::cout << "Splitted " << N_node << " data points: Points to the left: " << N_l << " points to the right: " << N_r << std::endl;
	node_t* node_l(new node_t);
	node_l->set_data(data_l);
	node_t* node_r(new node_t);
	node_r->set_data(data_r);
	return std::array<node_t*, 2>{ {node_l, node_r} };  
}

std::array<node_t*, 2> DensityTreeClassifier::split_node_gradient(node_t * node)
{
// get number of instances in this node
	size_t N_node = node->get_data().size1();
	double best_gradient = 0.;
	size_t best_instance = 0;
	size_t best_dim		 = 0;
// iterate over the instances
	for( size_t i = 0; i < N_node; i++ )
	{
// center the data around the i-th instance and then calculate its norm
		matrix<double> data( node->get_data() );
		matrix_row<matrix<double> > instance(data,i);
		vector<double> data_norm(N_node);
		for( size_t j = 0; j < N_node; j++ )
		{
			matrix_row<matrix<double> > instance_j(data,j);
// center around current instance
			instance_j -= instance;
// calculate the norm
			data_norm(j) = norm_2(instance_j);
		}
// sort-indices to find the k - nearest neighbors
		std::vector<size_t> sorted_indices = get_sorted_indices(data_norm);
// calculate the gradient and the density approximations
// the density is approximated by the mean distance to the k - nearest neighbors
		double density  = 0.;
// the gradient is approximated by the displacement of the instance to the cms of its k-nearest neighbors
		vector<double> cms( data.size2() );
		size_t k = mNearest_neighbors;
		if( k >= N_node )
		{
			k = N_node - 1;
		}
		for( size_t j = 1; j < k+1; j++)
		{
			size_t indx = sorted_indices[j];
			density += data_norm(indx);
			matrix_row<matrix<double> > instance_j(data,j);
			cms += instance_j;
		}
		density /= k;
		cms /= k;
		double gradient = norm_2(cms) / density;
		if( gradient > best_gradient)
		{
// find the dimension where the gradient changes the most
			auto iter = std::max_element( cms.begin(), cms.end() );
			size_t split_dim  = std::distance( cms.begin(), iter );
// make sure that this split is not trivial
			matrix_column<matrix<double> const> data_dim(node->get_data(), split_dim);
			vector<double> data_sort( data_dim.size() );
			std::copy( data_dim.begin(), data_dim.end(), data_sort.begin() );
			std::sort( data_sort.begin(), data_sort.end() );
			auto split_iter = std::lower_bound( data_sort.begin(), data_sort.end(), node->get_data()(i,split_dim) -0.01 );
			size_t N_l 		= std::distance(data_sort.begin(), split_iter);
			size_t N_r		= N_node - N_l;
			if(N_l >= 1 && N_r >= 1)
			{
				//std::cout << "New best gradient: " << gradient << " old: " << best_gradient << " instance: " << i << std::endl;
				best_gradient = gradient;
				best_instance = i;
				best_dim = split_dim;
			}
		}
	}
// store dimension and threshold in the node	
	node->set_split_dimension(best_dim);
// TODO where do we set the threshold, right or left of the split point ???
	double best_thresh = node->get_data()(best_instance,best_dim) - 0.01;
	node->set_split_threshold(best_thresh);
// split the data accordingly	
	matrix_column<matrix<double> const> data_dim( node->get_data(), best_dim );
	vector<double> data_sort( data_dim.size() );
	std::copy( data_dim.begin(), data_dim.end(), data_sort.begin() );
	std::sort( data_sort.begin(), data_sort.end() );
	auto split_iter = std::lower_bound(data_sort.begin(), data_sort.end(), best_thresh);
	size_t N_l 		= std::distance(data_sort.begin(), split_iter);
	size_t N_r		= N_node - N_l;
	image_data_t data_l( N_l, mNum_dimensions);
	image_data_t data_r( N_r, mNum_dimensions);
	size_t count_l = 0;
	size_t count_r = 0;
	for( size_t i = 0; i < N_node; i++)
	{
		if( node->get_data()(i,best_dim) < best_thresh ) // datapoint is on the left
		{
			matrix_row<matrix<double> const> 	copy_source( node->get_data(), i );
			matrix_row<matrix<double> >			copy_target( data_l, count_l );
			std::copy( copy_source.begin(), copy_source.end(), copy_target.begin() );
			count_l++;
		}
		else	// datapoint is on the right
		{
			matrix_row<matrix<double> const> 	copy_source( node->get_data(), i );
			matrix_row<matrix<double> >			copy_target( data_r, count_r );
			std::copy( copy_source.begin(), copy_source.end(), copy_target.begin() );
			count_r++;
		}
	}
	std::cout << "Splitted " << N_node << " data points: Points to the left: " << N_l << " points to the right: " << N_r << std::endl;
	node_t* node_l(new node_t);
	node_l->set_data(data_l);
	node_t* node_r(new node_t);
	node_r->set_data(data_r);
	return std::array<node_t*, 2>{ {node_l, node_r} };  
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
			double q = p_r / (p_l + p_r);    //WARNING: unused variable q
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
