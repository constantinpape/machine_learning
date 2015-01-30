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

	image_data_t generate(const short N, const short label);

private:
// private data member
	bool 	mTrained;
	size_t  mNum_instances;
	size_t 	mNum_classes;
	size_t 	mNum_dimensions;
// 1 tree for every class
	std::vector<node_t> mTrees;
	std::vector<double>	mPriors;
// private functions
	bool terminate_num(const node_t & node, const size_t N_class);
	bool terminate_depth(const node_t & node);
	std::array<node_t, 2> split_node(const node_t & node);
};
