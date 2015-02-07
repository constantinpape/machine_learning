#pragma once

#include "BayesClassifier.h"
#include "DensityTreeClassifier.h"

class CopulaClassifier : public Classifier
{
public:
	CopulaClassifier();

	void train(const image_data_t & train_data, const label_data_t & train_label);
	
	label_data_t predict(const image_data_t & test_data);

	image_data_t generate(const size_t N, const short label);

	void set_maximal_depth(const size_t max_depth);
	size_t get_maximal_depth() const;
	
	void   set_nearest_neighbors(const size_t num_nbrs);
	size_t get_nearest_neighbors() const;
private:
	bool mTrained;
	size_t mNum_instances;
	size_t mNum_classes;
	size_t mNum_dimensions;
	std::vector<double>	mPriors;
// BayesClassifier
	BayesClassifier	mBayes;
// DensityTreeClassifier
	DensityTreeClassifier mDensityTree;
// private functions
	image_data_t get_copula(const image_data_t & data);
};
