#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <algorithm>

int main()
{
	using namespace boost::numeric::ublas;
	matrix<int> m(4,4);
	m(0,0) = 1;
	m(1,1) = 2;
	m(0,1) = 3;
	m(2,1) = 7;
	m(3,1) = 4;
	std::cout << m << std::endl;

	matrix_column< matrix<int> > col(m,1);
	vector< int > col_copy( col.size() );
	std::copy( col.begin(), col.end(), col_copy.begin() );	

	std::cout << col_copy << std::endl;
	std::sort(col_copy.begin(), col_copy.end());
	std::cout << col_copy << std::endl;	
	std::cout << col << std::endl;	

	std::cout << "minmax" << std::endl;

	std::cout << *col_copy.begin() << std::endl;

	std::cout << *(col_copy.end()-1) << std::endl;

	//std::cout << m << std::endl;
	return 0;
}
