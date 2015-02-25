#include "../source/CopulaClassifier.h"

using namespace boost::numeric::ublas;

int main()
{
	CopulaClassifier copula_classifier;

	matrix<double> data(9,3);
	matrix<double> copula_expected(9,3);
// column 1 in order
	for(size_t i = 0; i < 9; i++)
	{
		data(i,0) = i;
		copula_expected(i,0) = (i+1.) / 10.;
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
	if( boost::numeric::ublas::detail::equals(copula_actual, copula_expected, 1.e-6, 0.) )
	{
		std::cout << "Test successfull" << std::endl;
	}
	else
	{
		std::cout << " Test not successfull: " << "\n" << " data_obtained = " << std::endl;
		std::cout << copula_actual << std::endl;
		std::cout <<" data_expected = " << std::endl;
		std::cout << copula_expected << std::endl;
	}
	return 0;
}
