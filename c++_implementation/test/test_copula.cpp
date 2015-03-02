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
// mnist: set bit for generating data to true, if data has full dimension
// init copula classifier and test it
	CopulaClassifier copula_classifier;
	//copula_classifier.set_maximal_depth(10);
	//copula_classifier.set_nearest_neighbors(5);
	//copula_classifier.set_split(DensityTreeClassifier::split_t::gradient);
	copula_classifier.set_record_split(true);
	test_classifier(copula_classifier, train_data, train_label, test_data, argv[4]);
	return 0;
}
