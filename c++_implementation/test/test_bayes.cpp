#include "../source/BayesClassifier.h"
#include "test_template.h"

int main(int argc, char* argv[])
{
	if( argc != 5)
	{
		std::cerr << "Need 4 inputs: Trainingdata, Traininglabel, Testdata, Filename for results" << std::endl;
		return 1;
	}
// read in the data
	image_data_t train_data  = read_mnist_data( argv[1] );
	label_data_t train_label = read_mnist_label(argv[2] );
	image_data_t test_data   = read_mnist_data( argv[3] );
// mnist: set bit for generating data to true, if data has full dimension
// scowl: set generating flag to true
// NOTE: yet no need to invoke parser library, I guess
	bool gen = true;
	if(train_data.size2() == 81)
	{
		gen = true;
	}
// init Bayes classifier and test it
	BayesClassifier bayes;
	test_classifier(bayes, train_data, train_label, test_data, argv[4], gen);
	return 0;
}
