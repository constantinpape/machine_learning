#pragma once

#include "utility/common.h"

class Classifier
{
public:
	Classifier()
	{}

	virtual void train(const image_data_t & train_data, const label_data_t & train_label) = 0;
	
	virtual label_data_t predict(const image_data_t & test_data) = 0;

	virtual image_data_t generate(const size_t N, const short label) = 0;
};
