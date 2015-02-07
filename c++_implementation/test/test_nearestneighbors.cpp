#include "../source/CopulaClassifier.h"

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
// init classifier	
	CopulaClassifier classifier;
	std::vector<size_t> k_values{ {5,10,15,30} };
	std::string fpath = argv[4];
// iterate over the depths and test for each
	for(size_t k : k_values)
	{
		classifier.set_nearest_neighbors(k);
		classifier.train(train_data, train_label);
		label_data_t results = classifier.predict(test_data);
		std::string res_name = fpath + "_results_" + std::to_string(k);
		save_label(res_name, results);
	}
	return 0;
}
