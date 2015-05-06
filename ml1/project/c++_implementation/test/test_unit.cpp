#include "gtest/gtest.h"
#include "../source/CopulaClassifier.h"

#include <string>
#include <boost/numeric/ublas/detail/matrix_assign.hpp>
using namespace boost::numeric::ublas;

TEST(CopulaFunctionality,RankOrderTransformation)
{
	CopulaClassifier copula_classifier;

	matrix<double> data(9,3);
	matrix<double> copula_expected(9,3);
// column 1 in order
	for(size_t i = 0; i < 9; i++)
	{
		data(i,0) = i;
		copula_expected(i,0) = (i + 1) / 10.;
	}
// column 2 in reverse order
	for(size_t i = 0; i < 9; i++)
	{
		data(i,1) = 8 - i;
		copula_expected(i,1) = (9 - i) / 10.;
	}
// column 3 in random order
	data(0,2) = 5;
	copula_expected(0,2) = 6 / 10.;
	data(1,2) = 3;
	copula_expected(1,2) = 4 / 10.;
	data(2,2) = 1;
	copula_expected(2,2) = 2 / 10.;
	data(3,2) = 8;
	copula_expected(3,2) = 9 / 10.;
	data(4,2) = 7;
	copula_expected(4,2) = 8 / 10.;
	data(5,2) = 0;
	copula_expected(5,2) = 1 / 10.;
	data(6,2) = 4;
	copula_expected(6,2) = 5 / 10.;
	data(7,2) = 2;
	copula_expected(7,2) = 3 / 10.;
	data(8,2) = 6;
	copula_expected(8,2) = 7 / 10.;
// get copula from the classifier
	matrix<double> copula_actual = copula_classifier.get_copula(data);
// check for equality
	for( size_t i = 0; i < data.size2(); i++ )
	{
		matrix_column<matrix<double> > view_actual(copula_actual,i);
		matrix_column<matrix<double> > view_expected(copula_expected,i);
		EXPECT_TRUE( detail::equals(view_actual, view_expected, 1.e-6, 0.) );
	}
}

//TODO test something here
TEST(CommonFuncionality,Data)
{
	std::string path = "~/machine_learning/c++_implementation/mnist_data";
// read in the data
	image_data_t train_data  = read_mnist_data( path + "/original/images_train.out");
	label_data_t train_label = read_mnist_label(path + "/original/labels_train.out");
	image_data_t test_data   = read_mnist_data( path + "/original/images_test.out");
	label_data_t test_label  = read_mnist_label(path + "/original/labels_test.out");
// assert that number of instances and dimensions do match	
	EXPECT_EQ( train_data.size1(), train_label.size() );
	EXPECT_EQ( test_data.size1() , test_label.size()  );
	EXPECT_EQ( train_data.size2(), test_data.size2()  );
// check data
	//std::cout << "Train-instances: " << train_data.size1() << " Dimensions: " << train_data.size2() << std::endl;
	//std::cout << "Test-instances: " << test_data.size1() << " Dimensions: " << test_data.size2() << std::endl;
	//for(int i = 0; i < train_data.size2(); ++i)
	//	std::cout << train_data(5,i) << std::endl;
	//for(int i = 0; i < 10; ++i)
	//	std::cout << train_label[i] << std::endl;
// save data and check that correct picture is reproduced (check is done with compare.py)
	save_data( path + "/test_readin/images_train.out", train_data);
	save_label(path + "/test_readin/labels_train.out", train_label);
}

TEST(CommonFuntionality,Sortindex)
{
	matrix<double> m(5,3);
	
	vector<double> order1(5);
	vector<double> order2(5);
	vector<double> order3(5);

	m(0,0) = 0;
	m(1,0) = 1;
	m(2,0) = 2;
	m(3,0) = 3;
	m(4,0) = 4;
	
	order1(0) = 0;
	order1(1) = 1;
	order1(2) = 2;
	order1(3) = 3;
	order1(4) = 4;
	
	m(0,1) = 4;
	m(1,1) = 3;
	m(2,1) = 2;
	m(3,1) = 1;
	m(4,1) = 0;
	
	order2(0) = 4;
	order2(1) = 3;
	order2(2) = 2;
	order2(3) = 1;
	order2(4) = 0;
	
	m(0,2) = 3;
	m(1,2) = 0;
	m(2,2) = 9;
	m(3,2) = 8;
	m(4,2) = 17;
	
	order3(0) = 1;
	order3(1) = 0;
	order3(2) = 3;
	order3(3) = 2;
	order3(4) = 4;

	matrix_column<matrix<double> const> view_1(m,0);
	matrix_column<matrix<double> const> view_2(m,1);
	matrix_column<matrix<double> const> view_3(m,2);
	
	vector<double> sorted1 = get_sorted_indices(view_1);
	vector<double> sorted2 = get_sorted_indices(view_2);
	vector<double> sorted3 = get_sorted_indices(view_3);
	
	EXPECT_TRUE( detail::equals( order1, sorted1, 1.e-6, 0.) );
	EXPECT_TRUE( detail::equals( order2, sorted2, 1.e-6, 0.) );
	EXPECT_TRUE( detail::equals( order3, sorted3, 1.e-6, 0.) );
}

int main(int argc, char **argv) 
{
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
