#include "test_template.h"

void test_classifier(Classifier & classifier,
		const image_data_t & train_data,
		const label_data_t & train_label,
		const image_data_t & test_data,
		const std::string & fpath)
{
// train the classifier
	classifier.train(train_data, train_label);
// predict the test data
	label_data_t results = classifier.predict(test_data);
	std::string fname = fpath + "/test_results.out";
	save_label(fname, results);
// TODO generate data
}
