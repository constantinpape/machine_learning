#include <algorithm>
#include <chrono>
#include <iterator>
#include <random>

#include "DensityTreeClassifier.h"

using namespace boost::numeric::ublas;
	
DensityTreeClassifier::DensityTreeClassifier() : 	mTrained(false),
													mNum_instances(0),
												 	mNum_classes(0),
													mNum_dimensions(0),
													mDepth_max(4),		// 4 == default value for maximal depth of the tree
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
// calculate the prior for this class
		mPriors.push_back( N_class/static_cast<double>(mNum_instances) );
// extract the data belonging to this class
		image_data_t data_class(N_class, mNum_dimensions);
		size_t class_count = 0;
		for(size_t i = 0; i < mNum_instances; i++)
		{
// copy the data if it has the correct class
			if( train_label[i] == c )
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
		std::vector<node_t> stack;
// initialize the root node
		node_t root;
		root.set_data(data_class);
		stack.push_back(root);
		size_t count = 0;
// build the tree
		while( !stack.empty() )
		{
// pop the last node from the stack
			node_t curr_node = *(stack.end() - 1);
			stack.pop_back();
// check whether this node is terminal - TODO try different termination criterion, currently using depth criterion
			//if( terminate_num(curr_node, N_class) )
			if( terminate_depth(curr_node) )
			{
				curr_node.calculate_probability(N_class);
				curr_node.set_terminal(true);
			}
// esle split the node, assign the children nodes and put them on the stack
			else
			{
				std::array<node_t, 2> children = split_node(curr_node); 	// TODO try different split criterion
				node_t & child_left  = children[0];
				node_t & child_right = children[1];
				child_left.set_depth(curr_node.get_depth() + 1);
				child_right.set_depth(curr_node.get_depth() + 1);
				curr_node.add_child( child_left);
				curr_node.add_child( child_right);
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

// Termination criterion depending on number of instances in the node
bool DensityTreeClassifier::terminate_num(const node_t & node, const size_t N_class)
{
	size_t N_node = node.get_data().size1();
	size_t N_min  = std::cbrt(N_class);
	if( N_node > N_min )
	{
		return false;
	}
	return true;
}

// Termination criterion depending on depth of node
bool DensityTreeClassifier::terminate_depth(const node_t & node)
{
// terminate the node if it is empyt
	if( node.get_data().size1() == 0 )
	{
		return true;
	}
// check the max_depth-criterion
	if( node.get_depth() >= mDepth_max )
	{
		return true;
	}
	return false;
}

// calculate the gain of the current split
double DensityTreeClassifier::calc_gain(const node_t & node, const double threshold, const size_t N, const size_t dimension)
{
	matrix_column<matrix<double> const> data_dim( node.get_data(), dimension );
	size_t N_l = std::count_if(data_dim.begin(), data_dim.end(), LessThreshold(threshold)  );
	size_t N_r = std::count_if(data_dim.begin(), data_dim.end(), GreaterThreshold(threshold) );
// no splits with 0 or 1 datapoints, because this would lead to diverging gains
	if( N_l <= 1 || N_r <= 1)
	{
		return 0.;
	}
// calculate volumes
	double Vol = node.get_volume();
// diviide by the volume of this dimension
	double min_dim = *( std::min_element( data_dim.begin(), data_dim.end() ) );
	double max_dim = *( std::max_element( data_dim.begin(), data_dim.end() ) );
	Vol /= (max_dim - min_dim);
	double V_l = Vol * ( threshold - min_dim );
	double V_r = Vol * ( max_dim - threshold );
	return std::pow( (static_cast<double>(N_l) / N), 2 ) / V_l + std::pow( (static_cast<double>(N_r) / N), 2 ) / V_r; 
}

// split node and return the two children nodes
std::array<node_t, 2> DensityTreeClassifier::split_node(node_t & node)
{
	double eps 	= 0.01;
	size_t N_node = node.get_data().size1();
	std::vector<double> thresholds;
	std::vector<double> gains;
// iterate all dimensions to find the best possible split
	for( size_t d = 0; d < mNum_dimensions; d++)
	{
		std::vector<double> thresholds_dim;
		std::vector<double> gains_dim;
// sort the data in this dimension
		matrix_column<matrix<double> const> data_aux(node.get_data(), d);
		vector<double> data_dim( data_aux.size() );
		std::copy( data_aux.begin(), data_aux.end(), data_dim.begin() );
		std::sort( data_dim.begin(), data_dim.end() );
// iterate over the data and calculate the gain for every possible split
// TODO we need to speed this up somehow!
		for( size_t i = 0; i < N_node; i++)
		{
			if( i!= 0 ) // dont look left for the leftmost instance
			{
				double thresh 	= data_dim(i) - eps;
				double gain 	= calc_gain(node, thresh, N_node, d);
				thresholds_dim.push_back( thresh );
				gains_dim.push_back( gain );
			}
			if( i!= N_node ) // dont look right for the rightmost instance
			{
				double thresh 	= data_dim(i) + eps;
				double gain 	= calc_gain(node, thresh, N_node, d);
				thresholds_dim.push_back( thresh );
				gains_dim.push_back( gain );
			}
		}
// look for the best split in this dimension 
		auto max_gain_elem 	= std::max_element( gains_dim.begin(), gains_dim.end() );
		size_t max_index	= std::distance( gains_dim.begin(), max_gain_elem );
		thresholds.push_back( thresholds_dim[max_index] );
		gains.push_back( *max_gain_elem );
	}
// look for the overall best split	
	auto max_gain_elem 	= std::max_element( gains.begin(), gains.end() );
	size_t d_opt		= std::distance( gains.begin(), max_gain_elem );
	double thresh_opt	= thresholds[d_opt];
// store dimension and threshold in the node	
	node.set_split_dimension(d_opt);
	node.set_split_threshold(thresh_opt);
// split the data coordingly	
	matrix_column<matrix<double> const> data_dim( node.get_data(), d_opt );
	size_t N_l = std::count_if(data_dim.begin(), data_dim.end(), LessThreshold(thresh_opt) );
	size_t N_r = std::count_if(data_dim.begin(), data_dim.end(), GreaterThreshold(thresh_opt) );
	assert(N_l + N_r == N_node);
	image_data_t data_l( N_l, mNum_dimensions);
	image_data_t data_r( N_r, mNum_dimensions);
	size_t count_l = 0;
	size_t count_r = 0;
	for( size_t i = 0; i < N_node; i++)
	{
		if( node.get_data()(i,d_opt) < thresh_opt ) // datapoint is on the left
		{
			matrix_row<matrix<double> const> 	copy_source( node.get_data(), i );
			matrix_row<matrix<double> >			copy_target( data_l, count_l );
			std::copy( copy_source.begin(), copy_source.end(), copy_target.begin() );
			count_l++;
		}
		else	// datapoint is on the right
		{
			matrix_row<matrix<double> const> 	copy_source( node.get_data(), i );
			matrix_row<matrix<double> >			copy_target( data_r, count_r );
			std::copy( copy_source.begin(), copy_source.end(), copy_target.begin() );
			count_r++;
		}
	}
	std::cout << "Splitted " << N_node << " data points: Points to the left: " << N_l << " points to the right: " << N_r << std::endl;
	node_t node_l;
	node_l.set_data(data_l);
	node_t node_r;
	node_r.set_data(data_r);
	return std::array<node_t, 2>{ {node_l, node_r} };  
}
	
label_data_t DensityTreeClassifier::predict(const image_data_t & test_data)
{
	if( !mTrained )
	{
		throw std::runtime_error("DensityTreeClassifier::predict: called before calling train!");
	}
	label_data_t label_return;
	std::cout << "BBB" << std::endl;
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
	std::cout << "BBB" << std::endl;
	return label_return;
}

// search the tree for the leaf-node which has the data_point
node_t DensityTreeClassifier::search_tree(const vector<double> & data_point, const size_t c )
{
// get the root node of the tree belonging to this class
	node_t & curr_node = mTrees[c];
// walk the tree until we come to a terminal node	
	while( !curr_node.get_terminal() )
	{
		size_t dim 		= curr_node.get_split_dimension();
		double thresh 	= curr_node.get_split_threshold();
// look whether this data_point is left or right of the split boundary
		std::cout << dim << std::endl;
		if( data_point(dim) < thresh )
		{
			std::cout << "BBB" << std::endl;
			curr_node = curr_node.get_child( node_t::side_t::left );
			std::cout << "BBB" << std::endl;
		}
		else
		{
			std::cout << "BBB" << std::endl;
			curr_node = curr_node.get_child( node_t::side_t::right );
			std::cout << "BBB" << std::endl;
		}
	}
	return curr_node;
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
		node_t & curr_node = mTrees[label];
// walk the tree until we get to a leaf-node
		while( !curr_node.get_terminal() )
		{
			size_t N = curr_node.get_data().size1();
			node_t & l_node = curr_node.get_child( node_t::side_t::left );
			node_t & r_node = curr_node.get_child( node_t::side_t::right);
// calculate p_left
			size_t N_l = l_node.get_data().size1();
			double V_l = l_node.get_volume();
			double p_l = N_l / (N * V_l);
// calculate p_right
			size_t N_r = r_node.get_data().size1();
			double V_r = r_node.get_volume();
			double p_r = N_r / (N * V_r);
// calculate p and q (normalised probabilities)
			double p = p_l / (p_l + p_r);
			double q = p_r / (p_l + p_r);
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
		const image_data_t & sample_data = curr_node.get_data();
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
	
void DensityTreeClassifier::set_maximal_depth(const size_t max_depth)
{
	mDepth_max = max_depth;
}

size_t DensityTreeClassifier::get_maximal_depth() const
{
	return mDepth_max;
}
