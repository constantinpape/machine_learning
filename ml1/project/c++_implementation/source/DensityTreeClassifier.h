#pragma once

#include <array>

#include "Classifier.h"
#include "utility/node_t.h"
#include "utility/splits.h"

class DensityTreeClassifier : public Classifier
{
public:
	enum split_t { def, def_alt, gradient, graph};
	DensityTreeClassifier();
	
	void train(const image_data_t & train_data, const label_data_t & train_label);
	
	label_data_t predict(const image_data_t & test_data);

	image_data_t generate(const size_t N, const short label);

	double get_likelihood(const ublas::vector<double> & data, const short label);

	void 	set_split(const split_t split);
	split_t get_split() const;

	void   set_maximal_depth(const size_t max_depth);
	size_t get_maximal_depth() const;
	
	void   set_nearest_neighbors(const size_t num_nbrs);
	size_t get_nearest_neighbors() const;

	void   set_shuffle(const bool enable, const size_t num_shuffle);

	void set_record_split(const bool enable);
	bool get_record_split() const;
	
	void set_radius(const double radius);
	double get_radius() const;
	
	void set_discrete_features(const size_t num_features);
	std::vector<double> get_feature_space() const;

private:
// private data member
	bool 	mTrained;
	size_t  mNum_instances;
	size_t 	mNum_classes;
	size_t 	mNum_dimensions;
// in case of discrete feature space, e.g alphabet
// NOTE: not usefull when using rank order, i mean copula
	size_t	mNum_features;
	std::vector<double> mFeature_space;
	size_t  mDepth_max;
	split_t mSplits;
	bool 	mRecord_splits;
	bool 	mDim_shuffle;
	size_t  mNum_shuffle;
	size_t  mNearest_neighbors;
	double  mRadius;
// 1 tree for every class
	std::vector<node_t*> 	mTrees;
	std::vector<double>	mPriors;
// private functions
	bool terminate_depth(const node_t * node);
	node_t search_tree(
			const ublas::vector<double> & data_point,
			const size_t c );
};
