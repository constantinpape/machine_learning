#include <chrono>
#include <iostream>

#include "test_template.h"

using namespace std::chrono;

void test_classifier(Classifier & classifier,
		const image_data_t & train_data,
		const label_data_t & train_label,
		const image_data_t & test_data,
		const std::string & fpath)
{
// train the classifier
	high_resolution_clock::time_point t_0 = high_resolution_clock::now();
	classifier.train(train_data, train_label);
	high_resolution_clock::time_point t_1 = high_resolution_clock::now();
// predict the test data
	label_data_t results = classifier.predict(test_data);
	high_resolution_clock::time_point t_2 = high_resolution_clock::now();
	save_label(fpath, results);
// generate 50 instances of the class 3 ( == 0 ) 
	unsigned int N = 500;
	high_resolution_clock::time_point t_3 = high_resolution_clock::now();
	image_data_t data_generated = classifier.generate(N,1);
	high_resolution_clock::time_point t_4 = high_resolution_clock::now();
	std::string gen_name = fpath + "_generated";
	save_data(gen_name, data_generated);
// report the train, prediction and generation time
	auto train_time = std::chrono::duration_cast<std::chrono::microseconds>( t_1 - t_0 ).count();
	auto predict_time = std::chrono::duration_cast<std::chrono::microseconds>( t_2 - t_1 ).count();
	auto gen_time = std::chrono::duration_cast<std::chrono::microseconds>( t_4 - t_3 ).count();
	std::cout << "Training took " << train_time << " microseconds" << std::endl;
	std::cout << "Predicting took " << predict_time << " microseconds" << std::endl;
	std::cout << "Generating took " << gen_time << " microseconds" << std::endl;
}
