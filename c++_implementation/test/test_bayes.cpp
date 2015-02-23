#include "../source/BayesClassifier.h"
#include "test_template.h"

int main(int argc, char* argv[])
{
	if ( (argc < 5) || (argc > 6) )
	{
		std::cerr << "4 input paths, 1 option at the end: trainingdata, traininglabel, "
			  << "testdata, filename for results [-g] (set generation on)" << std::endl;
		return 1;
	}
// check for command line flag for generation
// NOTE: yet no need to invoke parser library
	bool gen = false;
	if (argc == 6)
	{
		if (std::string(argv[5]) == "-g")         
	            	gen = true;
		}
	}
// read in the data
	image_data_t train_data  = read_mnist_data( argv[1] );
	label_data_t train_label = read_mnist_label(argv[2] );
	image_data_t test_data   = read_mnist_data( argv[3] );
// set bit for generating data to true, if data has full dimension
	if(train_data.size2() == 81)
	{
		gen = true;
	}
// init Bayes classifier and test it
	BayesClassifier bayes;
	test_classifier(bayes, train_data, train_label, test_data, argv[4], gen);
	return 0;
}
