#include <string>

#include "../source/common.h"
#include "../source/Classifier.h"

void test_classifier(Classifier & classifier,
		const image_data_t & train_data,
		const label_data_t & train_label,
		const image_data_t & test_data,
		const std::string & fpath);
