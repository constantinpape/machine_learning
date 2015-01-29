#include "../source/common.h"

int main()
{
// read in the data
	image_data_t train_data  = read_mnist_data( "../mnist_data/original/images_train.out");
	label_data_t train_label = read_mnist_label("../mnist_data/original/labels_train.out");
	image_data_t test_data   = read_mnist_data( "../mnist_data/original/images_test.out");
	label_data_t test_label  = read_mnist_label("../mnist_data/original/labels_test.out");
// assert that number of instances and dimensions do match	
	assert( train_data.size1() == train_label.size() );
	assert( test_data.size1()  == test_label.size()  );
	assert( train_data.size2() == test_data.size2()  );
// check data
	//std::cout << "Train-instances: " << train_data.size1() << " Dimensions: " << train_data.size2() << std::endl;
	//std::cout << "Test-instances: " << test_data.size1() << " Dimensions: " << test_data.size2() << std::endl;
	//for(int i = 0; i < train_data.size2(); ++i)
	//	std::cout << train_data(5,i) << std::endl;
	//for(int i = 0; i < 10; ++i)
	//	std::cout << train_label[i] << std::endl;
// save data and check that correct picture is reproduced (check is done with compare.py)
	save_data( "../mnist_data/test_readin/images_train.out", train_data);
	save_label("../mnist_data/test_readin/labels_train.out", train_label);
	return 0;
}
