#pragma once

#include "Classifier.h"

class BayesClassifier : public Classifier
{
public:
	BayesClassifier();
	
	void train(const image_data_t & train_data, const label_data_t & train_label);
	
	label_data_t predict(const image_data_t & test_data);

	image_data_t generate(const size_t N, const short label);

	ublas::vector<double> inverse_cdf(
			const ublas::matrix_row<ublas::matrix<double> > & data, const short label);

	double get_likelihood(
			const ublas::matrix_row<ublas::matrix<double> const> & data, const short label);

	ublas::vector<double> get_cdf(
			const ublas::matrix_row<ublas::matrix<double> const> & data, const short label);

	void set_fixed_bins(const bool enable, const size_t n);
	size_t get_fixed_bins() const;
	
private:
	//private variables
	bool trained;
	bool fixed_bins;
	size_t num_instances;
	size_t num_dimensions;
	size_t num_classes;
	size_t num_fixed_bins;
	std::vector<size_t>		instances_per_class;
	std::vector<double>     priors;
	std::vector<bin_t> 		bins;
	std::vector<size_t> 	irrelevant_dims;
	std::vector<histogram_t> histograms;
	std::vector<cdf_t >		cdfs;					
	//private functions
	bin_t get_optimal_bins(const image_data_t & train_data, const size_t dim);
	bin_t get_fixed_bins();
	void compute_cdf();
};
