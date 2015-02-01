// test for the function get_sorted_indices from common 
// TODO test corner cases

#include "../source/common.h"

using namespace boost::numeric::ublas;

bool vector_equal(const vector<double> & lhs, const vector<double> & rhs)
{
	if( lhs.size() != rhs.size() )
		return false;
	for(size_t i = 0; i < lhs.size(); i++)
	{
		if( lhs(i) != rhs(i) )
			return false;
	}
	return true;
}

int main()
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
	if( vector_equal(order1, sorted1) )
	{
		std::cout << "True" << std::endl;
	}
	else
	{
		std::cout << "False" << std::endl;
		std::cout << "Expected: " << order1 << std::endl;
		std::cout << "Actual: " << sorted1 << std::endl;
	}
	
	vector<double> sorted2 = get_sorted_indices(view_2);
	if( vector_equal(order2, sorted2) )
	{
		std::cout << "True" << std::endl;
	}
	else
	{
		std::cout << "False" << std::endl;
		std::cout << "Expected: " << order2 << std::endl;
		std::cout << "Actual: " << sorted2 << std::endl;
	}
	
	vector<double> sorted3 = get_sorted_indices(view_3);
	if( vector_equal(order3, sorted3) )
	{
		std::cout << "True" << std::endl;
	}
	else
	{
		std::cout << "False" << std::endl;
		std::cout << "Expected: " << order3 << std::endl;
		std::cout << "Actual: " << sorted3 << std::endl;
	}

	return 0;
}
