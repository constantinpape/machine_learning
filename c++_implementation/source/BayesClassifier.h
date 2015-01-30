#pragma once

#include "common.h"
#include "Classifier.h"

class BayesClassifier : public Classifier
{
public:
	BayesClassifier();
	
	void train(const image_data_t & train_data, const label_data_t & train_label);
	
	label_data_t predict(const image_data_t & test_data);

	image_data_t generate(const short N, const short label);
	
	void compute_cdf();
	
private:
	//private variables
	bool trained;
	size_t num_instances;
	size_t num_dimensions;
	size_t num_classes;
	std::vector<size_t>		instances_per_class;
	std::vector<double>     priors;
	std::vector<bin_t> 		bins;
	std::vector<size_t> 	irrelevant_dims;
	std::vector<histogram_t> histograms;
	//private functions
	bin_t get_optimal_bins(const image_data_t & train_data, const size_t dim);
};
