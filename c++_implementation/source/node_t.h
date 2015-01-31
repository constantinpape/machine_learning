#pragma once

#include <vector>

#include "common.h"

class node_t
{
public:
// enum for left / right
	enum side_t { left = 0, right = 1};
// public functions
	node_t();
	void 	set_data(const image_data_t & data);
	const image_data_t & get_data() const;
	void	add_child(node_t* child);
	const node_t & get_child( const side_t side ) const;
	node_t & get_child( const side_t side );
	void 	calculate_probability( const size_t N_class );
	double 	get_probability() const;
	double 	get_volume() const;
	void	set_depth(const size_t depth);
	size_t	get_depth() const;
	void 	set_split_dimension(const size_t dim);
	size_t 	get_split_dimension() const;
	void 	set_split_threshold(const double thresh);
	double 	get_split_threshold() const;
	void 	set_terminal(const bool terminal);
	bool 	get_terminal() const;
// static member
	static const size_t max_nodes = 2;
private:
// private data
	image_data_t 	mData;
	std::vector<node_t*> mChildren;	// store the 2 childeren of this node 0 -> left child, 1 -> right child
	bool 			mTerminal;
	double 			mProbability;
	double 			mVolume;
	size_t 			mDepth;
	size_t 			mSplit_dimension;
	double 			mSplit_threshold;
// private functions
	void 	calculate_volume(); // this is called whithin set_data()
};
