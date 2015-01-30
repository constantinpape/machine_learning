#include <algorithm>
#include <iterator>

#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>

#include "DensityTreeClassifier.h"

using namespace boost::numeric::ublas;
	
DensityTreeClassifier::DensityTreeClassifier() : 	mTrained(false),
													mNum_instances(0),
												 	mNum_classes(0),
													mNum_dimensions(0),
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
// number of instances beloniging to this class
		size_t N_class = std::count( train_label.begin(), train_label.end(), c);
// calculate the prior for this class
		mPriors.push_back( N_class/static_cast<double>(mNum_instances) );
// extract the data belonging to this class
		image_data_t data_class(N_class, mNum_dimensions);
		size_t class_count = 0;
		for(auto it = train_label.begin(); it != train_label.end(); it++)
		{
// copy the data if it has the correct class
			if( *it == c )
			{
				size_t index = std::distance(train_label.begin(),it);
				matrix_row<matrix<double> const> orig_instance(train_data, index);
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
				curr_node.add_child( new node_t(child_left)  );
				curr_node.add_child( new node_t(child_right) );
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

// Termination criterion depending on number of instances in the node TODO implement
bool DensityTreeClassifier::terminate_num(const node_t & node, const size_t N_class)
{

}

// Termination criterion depending on depth of node TODO implement
bool DensityTreeClassifier::terminate_depth(const node_t & node)
{

}

// split node and return the two children nodes TODO implement
std::array<node_t, 2> DensityTreeClassifier::split_node(const node_t & node)
{

}
	
label_data_t DensityTreeClassifier::predict(const image_data_t & test_data)
{
	label_data_t label_return;

	return label_return;
}

image_data_t DensityTreeClassifier::generate(const short N, const short label)
{
	image_data_t data_return;

	return data_return;
}
