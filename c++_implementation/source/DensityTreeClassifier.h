#pragma once

#include <array>

#include "Classifier.h"
#include "node_t.h"

class DensityTreeClassifier : public Classifier
{
public:
	DensityTreeClassifier();
	
	void train(const image_data_t & train_data, const label_data_t & train_label);
	
	label_data_t predict(const image_data_t & test_data);

	image_data_t generate(const size_t N, const short label);

	double get_likelihood(const boost::numeric::ublas::vector<double> & data, const short label);

	void   set_maximal_depth(const size_t max_depth);
	size_t get_maximal_depth() const;
	
	void   set_nearest_neighbors(const size_t num_nbrs);
	size_t get_nearest_neighbors() const;

	void   set_shuffle(const bool enable, const size_t num_shuffle);

private:
// private data member
	bool 	mTrained;
	bool 	mDim_shuffle;
	size_t  mNum_instances;
	size_t 	mNum_classes;
	size_t 	mNum_dimensions;
	size_t  mDepth_max;
	size_t  mNum_shuffle;
	size_t  mNearest_neighbors;
// 1 tree for every class
	std::vector<node_t*> mTrees;
	std::vector<double>	mPriors;
// private functions
	bool terminate_depth(const node_t * node);
	std::array<node_t*, 2> split_node(node_t * node, const size_t N_min);
	std::array<node_t*, 2> split_node_gradient(node_t * node);
	std::array<node_t*, 2> split_node_dimshuffle(node_t * node, const size_t N_min, const size_t dims);
	node_t search_tree(
			const boost::numeric::ublas::vector<double> & data_point,
			const size_t c );
};
