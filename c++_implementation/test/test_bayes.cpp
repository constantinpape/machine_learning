#include "../source/BayesClassifier.h"
#include "test_template.h"

int main()
{
// read in the data
	image_data_t train_data  = read_mnist_data( "../mnist_data/original/images_train.out");
	label_data_t train_label = read_mnist_label("../mnist_data/original/labels_train.out");
	image_data_t test_data   = read_mnist_data( "../mnist_data/original/images_test.out");
// init Bayes classifier and test it
	BayesClassifier bayes;
	std::string filepath("../mnist_data/test_bayes");
	test_classifier(bayes, train_data, train_label, test_data, filepath);
	return 0;
}
