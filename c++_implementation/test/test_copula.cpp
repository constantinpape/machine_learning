#include "../source/CopulaClassifier.h"
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
// set bit for generating data to true, if data has full dimension
	bool gen = false;
	if( train_data.size2() == 81)
	{
		gen = true;
	}
// init Bayes classifier and test it
	CopulaClassifier copula_classifier;
	classifier.set_maximal_depth(5);
	classifier.set_nearest_neighbors(5);
	test_classifier(copula_classifier, train_data, train_label, test_data, argv[4], gen);
	return 0;
}
