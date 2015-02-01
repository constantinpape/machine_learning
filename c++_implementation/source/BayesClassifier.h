#pragma once

#include "Classifier.h"

class BayesClassifier : public Classifier
{
public:
	BayesClassifier();
	
	void train(const image_data_t & train_data, const label_data_t & train_label);
	
	label_data_t predict(const image_data_t & test_data);

	image_data_t generate(const size_t N, const short label);

	boost::numeric::ublas::vector<double> inverse_cdf(
			const boost::numeric::ublas::matrix_row<boost::numeric::ublas::matrix<double> > & data, const short label);

	double get_likelihood(
			const boost::numeric::ublas::matrix_row<boost::numeric::ublas::matrix<double> const> & data, const short label);

	boost::numeric::ublas::vector<double> get_cdf(
			const boost::numeric::ublas::matrix_row<boost::numeric::ublas::matrix<double> const> & data, const short label);
	
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
	std::vector<cdf_t >		cdfs;					
	//private functions
	bin_t get_optimal_bins(const image_data_t & train_data, const size_t dim);
	void compute_cdf();
};
