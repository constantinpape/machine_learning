#include "test_template.h"

void test_classifier(Classifier & classifier,
		const image_data_t & train_data,
		const label_data_t & train_label,
		const image_data_t & test_data,
		const std::string & fpath,
		bool gen)
{
// train the classifier
	classifier.train(train_data, train_label);
// predict the test data
	label_data_t results = classifier.predict(test_data);
	save_label(fpath, results);
// generate 50 instances of the class 3 ( == 0 ) if gen == true 
	if( gen == true)
	{
		unsigned int N = 50;
		image_data_t data_generated = classifier.generate(N,0);
		std::string gen_name = fpath + "_generated";
		save_data(gen_name, data_generated);
	}
}
