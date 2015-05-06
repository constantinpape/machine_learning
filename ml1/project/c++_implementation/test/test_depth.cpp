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
// make sure data has full dimension
	assert( test_data.size2() == 81 );
// init Copula Classifier and test it
	CopulaClassifier classifier;
	std::vector<size_t> depths{ {4,8,10,20} };
	std::string fpath = argv[4];
// iterate over the depths and test for each
	for(size_t depth : depths)
	{
		classifier.set_maximal_depth(depth);
		classifier.train(train_data, train_label);
		label_data_t results = classifier.predict(test_data);
		image_data_t data_generated = classifier.generate(50,0);
		std::string res_name = fpath + "_results_" + std::to_string(depth);
		std::string gen_name = fpath + "_generated_" + std::to_string(depth);
		save_label(res_name, results);
		save_data(gen_name, data_generated);
	}
	return 0;
}
