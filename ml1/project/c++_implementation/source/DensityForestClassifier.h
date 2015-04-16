#pragma once

#include "Classifier.h"
#include "DensityTreeClassifier.h" 

class DensityForestClassifier : public Classifier
{
public:
	DensityForestClassifier();

	void train(const image_data_t & train_data, const label_data_t & train_label);
	
	label_data_t predict(const image_data_t & test_data);

	image_data_t generate(const size_t N, const short label);
	
	double get_likelihood(const boost::numeric::ublas::vector<double> & data, const short label);
	
	void   set_maximal_depth(const size_t max_depth);
	size_t get_maximal_depth() const;
	
	void   set_number_trees(const size_t num_trees);
	size_t get_number_trees() const;
	
	void   set_nearest_neighbors(const size_t num_nbrs);
	size_t get_nearest_neighbors() const;

private:
	bool mTrained;
	size_t mNumInstances;
	size_t mNumDimensions;
	size_t mNumClasses;
	size_t mNumTrees;
	std::vector<DensityTreeClassifier> mTrees;
};
