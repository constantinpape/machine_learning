#pragma once

#include <memory>

#include "common.h"
#include "Classifier.h"

// ForwardDeclarations
class BayesClassifier;
class DensityTreeClassifier;

class CopulaClassifier : public Classifier
{
public:
	CopulaClassifier();

	void train(const image_data_t & train_data, const label_data_t & train_label);
	
	label_data_t predict(const image_data_t & test_data);

	image_data_t generate(const size_t N, const short label);

	void set_maximal_depth(const size_t max_depth);
	size_t get_maximal_depth() const;
private:
	bool mTrained;
	size_t mNum_instances;
	size_t mNum_classes;
	size_t mNum_dimensions;
	std::vector<double>	mPriors;
// pointer to BayesClassifier
	std::unique_ptr<BayesClassifier>		mBayes;
// pointer to DensityTreeClassifier
	std::unique_ptr<DensityTreeClassifier>	mDensityTree;
// private functions
	image_data_t get_copula(const image_data_t & data);
};
