#pragma once

#include <array>
#include <boost/numeric/ublas/matrix_proxy.hpp>

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

private:
// private data member
	bool 	mTrained;
	size_t  mNum_instances;
	size_t 	mNum_classes;
	size_t 	mNum_dimensions;
	size_t  mDepth_max;
// 1 tree for every class
	std::vector<node_t> mTrees;
	std::vector<double>	mPriors;
// private functions
	bool terminate_num(const node_t & node, const size_t N_class);
	bool terminate_depth(const node_t & node);
	double calc_gain(const node_t & node, const double threshold, const size_t N, const size_t dimension);
	std::array<node_t, 2> split_node(node_t & node);
	node_t search_tree(const boost::numeric::ublas::vector<double> & data_point, const size_t c );
};
