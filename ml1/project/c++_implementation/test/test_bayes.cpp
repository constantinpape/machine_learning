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
// init Bayes classifier and test it
	BayesClassifier bayes;
	//bayes.set_fixed_bins(true, 26);
	test_classifier(bayes, train_data, train_label, test_data, argv[4]);
	return 0;
}
