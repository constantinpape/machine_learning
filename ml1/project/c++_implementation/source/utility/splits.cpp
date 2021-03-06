#include <fstream>
#include <iostream> 
#include <array>

#include "splits.h"

#include <boost/numeric/ublas/matrix_sparse.hpp>
//#include <boost/numeric/bindings/lapack/syev.hpp>

using namespace boost::numeric::ublas;
//using namespace boost::numeric::bindings::lapack;


// split node and return the two children nodes
std::array<node_t*, 2> split_node_default(
    node_t * node, const bool dim_shuffle, const size_t num_shuffle, const bool record)
{
// epsilon for the thresholds
	double eps 	= 0.01;
// get number of instances in this node
	size_t N_node = node->get_data().size1();
	size_t num_dimensions = node->get_data().size2();
	double best_thresh = 0.;
	double best_gain   = 1.; 
	size_t best_dim	   = 0;
	std::vector<size_t> dimensions;
	std::fstream stream;
	if( record )
	{
		std::string fname = "split" + std::to_string(node->get_depth());
		stream.open(fname, std::fstream::out);
		formatted_ostream output(stream, 10);
		output << "#gain" << "V_left" << "V_right" << "N_left" << "N_right" << std::endl;
	}
	for( size_t d = 0; d < num_dimensions; d++)
	{
		dimensions.push_back(d);
	}
// if mDim_shuffle == true shuffle the dimensions and only iterate over the first mNum_shuffle entries
	if( dim_shuffle )
	{
		std::random_device rd;
		std::mt19937 g( rd() );
		std::shuffle( dimensions.begin(), dimensions.end(), g );
		dimensions.resize(num_shuffle);
	}
// iterate all dimensions to find the best possible split
	for( size_t d : dimensions )
	{
// sort the data in this dimension
		matrix_column<matrix<double> const> data_aux(node->get_data(), d);
		vector<double> data_dim( data_aux.size() );
		std::copy( data_aux.begin(), data_aux.end(), data_dim.begin() );
		std::sort( data_dim.begin(), data_dim.end() );
		assert( data_dim.size() == N_node );
// precompute the volume in this dimension
		double V_dim = node->get_volume();
// divide by the volume of this dimension
		double min_thres = *( data_dim.begin() ) + eps;
		double max_thres = *( data_dim.end() - 1) - eps;
		double V_red = V_dim / ( *(data_dim.end()-1) - *(data_dim.begin()) );
// calculate all threshold
		std::vector<double> thresholds;
		for( size_t i = 0; i < N_node; i++ )
		{
			if (data_dim[i] - eps > min_thres )
			{
				thresholds.push_back(data_dim[i] - eps);
			}	 
			if( data_dim[i] + eps < max_thres )
			{
				thresholds.push_back(data_dim[i] + eps);
			}	 
		}
// iterate over the thresholds
		for( double t : thresholds)
		{
			auto split_iter = std::lower_bound(data_dim.begin(), data_dim.end(), t);
			size_t N_l 		= std::distance(data_dim.begin(), split_iter);
			size_t N_r 		= N_node - N_l;
// calculate volumes
			double V_l = V_red * ( t - min_thres );
			double V_r = V_red * ( max_thres - t );
			//if ( (t-min_thres <= eps) || (max_thres-t <= eps)){continue;}
			if ( (V_l == 0) || (V_r == 0) ){continue;}
			double gain_l = std::pow( (static_cast<double>(N_l) / N_node), 2 )*(V_dim / V_l);
			double gain_r = std::pow( (static_cast<double>(N_r) / N_node), 2 )*(V_dim / V_r);
			double gain = gain_l + gain_r; 
			if( record )
			{
				formatted_ostream output(stream, 10);
				output << gain << V_l << V_r << N_l << N_r << std::endl; 
			}
// check whether this is the best gain so far, but exclude too small splits
			if( gain > best_gain && N_l > 1 && N_r > 1 )
			{
				best_gain 	= gain;
				best_thresh = t;
				best_dim 	= d; 
			}
		}
		if( record )
		{
			stream << std::endl; 
		}
	}
// store dimension and threshold in the node	
	node->set_split_dimension(best_dim);
	node->set_split_threshold(best_thresh);
// split the data accordingly	
	matrix_column<matrix<double> const> data_dim( node->get_data(), best_dim );
	vector<double> data_sort( data_dim.size() );
	std::copy( data_dim.begin(), data_dim.end(), data_sort.begin() );
	std::sort( data_sort.begin(), data_sort.end() );
	auto split_iter = std::lower_bound(data_sort.begin(), data_sort.end(), best_thresh);
	size_t N_l 		= std::distance(data_sort.begin(), split_iter);
	size_t N_r		= N_node - N_l;
	image_data_t data_l( N_l, num_dimensions);
	image_data_t data_r( N_r, num_dimensions);
	size_t count_l = 0;
	size_t count_r = 0;
	for( size_t i = 0; i < N_node; i++)
	{
		if( node->get_data()(i,best_dim) < best_thresh ) // datapoint is on the left
		{
			matrix_row<matrix<double> const> 	copy_source( node->get_data(), i );
			matrix_row<matrix<double> >			copy_target( data_l, count_l );
			std::copy( copy_source.begin(), copy_source.end(), copy_target.begin() );
			count_l++;
		}
		else	// datapoint is on the right
		{
			matrix_row<matrix<double> const> 	copy_source( node->get_data(), i );
			matrix_row<matrix<double> >			copy_target( data_r, count_r );
			std::copy( copy_source.begin(), copy_source.end(), copy_target.begin() );
			count_r++;
		}
	}
	assert ( count_l == N_l );
	assert ( count_r == N_r );
	std::cout << "Splitted " << N_node << " data points: "
		  <<  N_l << " to the left "
		  <<  N_r << " to the right" << std::endl;
	node_t* node_l(new node_t);
	node_t* node_r(new node_t);
	if (node->get_discrete())
	{
		node_l->set_discrete(true);
		node_r->set_discrete(true);
	}
	node_l->set_data(data_l);
	node_r->set_data(data_r);
	return std::array<node_t*, 2>{ {node_l, node_r} };  
}


// split node and return the two children nodes
std::array<node_t*, 2> split_node_alt(node_t * node, const bool dim_shuffle, const size_t num_shuffle, const bool record)
{
// epsilon for the thresholds
	double eps 	= 0.01;
// get number of instances in this node
	size_t N_node = node->get_data().size1();
	size_t num_dimensions = node->get_data().size2();
	double best_thresh = 0.;
	double best_gain   = 0.; 
	size_t best_dim	   = 0;
	std::vector<size_t> dimensions;
	std::fstream stream;
	if( record )
	{
		std::string fname = "split" + std::to_string(node->get_depth());
		stream.open(fname, std::fstream::out);
		formatted_ostream output(stream, 10);
		output << "#gain" << "V_left" << "V_right" << "N_left" << "N_right" << std::endl;
	}
	for( size_t d = 0; d < num_dimensions; d++)
	{
		dimensions.push_back(d);
	}
// if mDim_shuffle == true shuffle the dimensions and only iterate over the first mNum_shuffle entries
	if( dim_shuffle )
	{
		std::random_device rd;
		std::mt19937 g( rd() );
		std::shuffle( dimensions.begin(), dimensions.end(), g );
		dimensions.resize(num_shuffle);
	}
// iterate all dimensions to find the best possible split
	for( size_t d : dimensions )
	{
// sort the data in this dimension
		matrix_column<matrix<double> const> data_aux(node->get_data(), d);
		vector<double> data_dim( data_aux.size() );
		std::copy( data_aux.begin(), data_aux.end(), data_dim.begin() );
		std::sort( data_dim.begin(), data_dim.end() );
		assert( data_dim.size() == N_node );
// precompute the volume in this dimension
		double V_dim = node->get_volume();
// divide by the volume of this dimension
		double min_dim = *( std::min_element( data_dim.begin(), data_dim.end() ) );
		double max_dim = *( std::max_element( data_dim.begin(), data_dim.end() ) );
		V_dim /= (max_dim - min_dim);
// calculate all threshold
		std::vector<double> thresholds;
		double min_thresh = *data_dim.begin() + eps;
		double max_thresh = *(data_dim.end()-1) - eps;
		for( size_t i = 1; i < N_node; i++ )
		{
			if( data_dim[i] - eps > min_thresh )
			{
				thresholds.push_back(data_dim[i] - eps);
			}	 
		}
		for( size_t i = 0; i < N_node; i++ )
		{
			if( data_dim[i] + eps < max_thresh )
			{
				thresholds.push_back(data_dim[i] + eps);
			}	 
		}
// iterate over the thresholds
		for( double t : thresholds)
		{
			auto split_iter = std::lower_bound(data_dim.begin(), data_dim.end(), t);
			size_t N_l 		= std::distance(data_dim.begin(), split_iter);
			size_t N_r 		= N_node - N_l;
// calculate volumes
			double V_l = V_dim * ( t - min_dim );
			double V_r = V_dim * ( max_dim - t );
			if ( (V_l == 0) || (V_r == 0) ){continue;}
			//double gain = fabs( std::pow(N_l * V_r,2) - std::pow(N_r * V_l,2)); 
			double gain = fabs( N_l * V_r - N_r * V_l); 
			if( record )
			{
				formatted_ostream output(stream, 10);
				output << gain << V_l << V_r << N_l << N_r << std::endl; 
			}
// check whether this is the best gain so far, but exclude too small splits
			if( gain > best_gain && N_l > 1 && N_r > 1 )
			{
				best_gain 	= gain;
				best_thresh = t;
				best_dim 	= d; 
			}
		}
		if( record )
		{
			stream << std::endl; 
		}
	}
// store dimension and threshold in the node	
	node->set_split_dimension(best_dim);
	node->set_split_threshold(best_thresh);
// split the data accordingly	
	matrix_column<matrix<double> const> data_dim( node->get_data(), best_dim );
	vector<double> data_sort( data_dim.size() );
	std::copy( data_dim.begin(), data_dim.end(), data_sort.begin() );
	std::sort( data_sort.begin(), data_sort.end() );
	auto split_iter = std::lower_bound(data_sort.begin(), data_sort.end(), best_thresh);
	size_t N_l 		= std::distance(data_sort.begin(), split_iter);
	size_t N_r		= N_node - N_l;
	image_data_t data_l( N_l, num_dimensions);
	image_data_t data_r( N_r, num_dimensions);
	size_t count_l = 0;
	size_t count_r = 0;
	for( size_t i = 0; i < N_node; i++)
	{
		if( node->get_data()(i,best_dim) < best_thresh ) // datapoint is on the left
		{
			matrix_row<matrix<double> const> 	copy_source( node->get_data(), i );
			matrix_row<matrix<double> >			copy_target( data_l, count_l );
			std::copy( copy_source.begin(), copy_source.end(), copy_target.begin() );
			count_l++;
		}
		else	// datapoint is on the right
		{
			matrix_row<matrix<double> const> 	copy_source( node->get_data(), i );
			matrix_row<matrix<double> >			copy_target( data_r, count_r );
			std::copy( copy_source.begin(), copy_source.end(), copy_target.begin() );
			count_r++;
		}
	}
	assert ( count_l == N_l );
	assert ( count_r == N_r );
	std::cout << "Splitted " << N_node << " data points: "
		  <<  N_l << " to the left "
		  <<  N_r << " to the right" << std::endl;
	node_t* node_l(new node_t);
	node_l->set_data(data_l);
	node_t* node_r(new node_t);
	node_r->set_data(data_r);
	return std::array<node_t*, 2>{ {node_l, node_r} };  
}


std::array<node_t*, 2> split_node_gradient(node_t * node, const size_t nearest_neighbors, const bool record)
{
// get number of instances in this node
	size_t N_node = node->get_data().size1();
	size_t num_dimensions = node->get_data().size2();
	double best_gradient = 0.;
	size_t best_instance = 0;
	size_t best_dim		 = 0;
	std::fstream stream;
	if( record )
	{
		std::string fname = "split" + std::to_string(node->get_depth());
		stream.open(fname, std::fstream::out);
		formatted_ostream output(stream, 10);
		output << "#gradient" << std::endl;
	}
// iterate over the instances
	for( size_t i = 0; i < N_node; i++ )
	{
// center the data around the i-th instance and then calculate its norm
		matrix<double> data( node->get_data() );
		matrix_row<matrix<double> > instance(data,i);
		vector<double> data_norm(N_node);
		for( size_t j = 0; j < N_node; j++ )
		{
			matrix_row<matrix<double> > instance_j(data,j);
// center around current instance
			instance_j -= instance;
// calculate the norm
			data_norm(j) = norm_2(instance_j);
		}
// rank indices to find the k - nearest neighbors
		vector<double> sorted_indices = get_ranked_indices( data_norm );
// calculate the gradient and the density approximations
// the density is approximated by the mean distance to the k - nearest neighbors
		double density  = 0.;
// the gradient is approximated by the displacement of the instance to the cms of its k-nearest neighbors
		vector<double> cms( data.size2() );
		size_t k = nearest_neighbors;
		if( k >= N_node )
		{
			k = N_node - 1;
		}
		for( size_t j = 1; j < k+1; j++)
		{
			size_t indx = static_cast<size_t>( sorted_indices(j) );
			density += data_norm(indx);
			matrix_row<matrix<double> > instance_j(data,j);
			cms += instance_j;
		}
		density /= k;
		cms /= k;
		double gradient = norm_2(cms) / density;
		if( record )
		{
			formatted_ostream output(stream, 10);
			output << gradient << std::endl; 
		}
		if( gradient > best_gradient)
		{
// find the dimension where the gradient changes the most
			auto iter = std::max_element( cms.begin(), cms.end() );
			size_t split_dim  = std::distance( cms.begin(), iter );
// make sure that this split is not trivial
			matrix_column<matrix<double> const> data_dim(node->get_data(), split_dim);
			vector<double> data_sort( data_dim.size() );
			std::copy( data_dim.begin(), data_dim.end(), data_sort.begin() );
			std::sort( data_sort.begin(), data_sort.end() );
			auto split_iter = std::lower_bound( data_sort.begin(), data_sort.end(), node->get_data()(i,split_dim) -0.01 );
			size_t N_l 		= std::distance(data_sort.begin(), split_iter);
			size_t N_r		= N_node - N_l;
			if(N_l >= 1 && N_r >= 1)
			{
				//std::cout << "New best gradient: " << gradient << " old: " << best_gradient << " instance: " << i << std::endl;
				best_gradient = gradient;
				best_instance = i;
				best_dim = split_dim;
			}
		}
	}
// store dimension and threshold in the node	
	node->set_split_dimension(best_dim);
// TODO where do we set the threshold, right or left of the split point ???
	double best_thresh = node->get_data()(best_instance,best_dim) - 0.01;
	node->set_split_threshold(best_thresh);
// split the data accordingly	
	matrix_column<matrix<double> const> data_dim( node->get_data(), best_dim );
	vector<double> data_sort( data_dim.size() );
	std::copy( data_dim.begin(), data_dim.end(), data_sort.begin() );
	std::sort( data_sort.begin(), data_sort.end() );
	auto split_iter = std::lower_bound(data_sort.begin(), data_sort.end(), best_thresh);
	size_t N_l 		= std::distance(data_sort.begin(), split_iter);
	size_t N_r		= N_node - N_l;
	image_data_t data_l( N_l, num_dimensions);
	image_data_t data_r( N_r, num_dimensions);
	size_t count_l = 0;
	size_t count_r = 0;
	for( size_t i = 0; i < N_node; i++)
	{
		if( node->get_data()(i,best_dim) < best_thresh ) // datapoint is on the left
		{
			matrix_row<matrix<double> const> 	copy_source( node->get_data(), i );
			matrix_row<matrix<double> >			copy_target( data_l, count_l );
			std::copy( copy_source.begin(), copy_source.end(), copy_target.begin() );
			count_l++;
		}
		else	// datapoint is on the right
		{
			matrix_row<matrix<double> const> 	copy_source( node->get_data(), i );
			matrix_row<matrix<double> >			copy_target( data_r, count_r );
			std::copy( copy_source.begin(), copy_source.end(), copy_target.begin() );
			count_r++;
		}
	}
	assert ( count_l == N_l );
	assert ( count_r == N_r );
	std::cout << "Splitted " << N_node << " data points: "
		  <<  N_l << " to the left "
		  <<  N_r << " to the right" << std::endl;
	node_t* node_l(new node_t);
	node_l->set_data(data_l);
	node_t* node_r(new node_t);
	node_r->set_data(data_r);
	return std::array<node_t*, 2>{ {node_l, node_r} };  
}


std::array<node_t*, 2> split_node_graph(node_t * node, const size_t max_radius, const bool record )
{
	size_t N_node = node->get_data().size1(); 
	std::fstream stream;
	if( record )
	{
		std::string fname = "split" + std::to_string(node->get_depth());
		stream.open(fname, std::fstream::out);
		formatted_ostream output(stream, 10);
		output << "#gain" << std::endl;
	}
// build the graph
// Adjacency matrix
	compressed_matrix<size_t> A( N_node, N_node );
// iterate over the instances
	for( size_t i = 0; i < N_node; i++ )
	{
// center the data around the i-th instance and then calculate its norm
		matrix<double> data( node->get_data() );
		matrix_row<matrix<double> > instance(data,i);
		vector<double> data_norm(N_node);
		for( size_t j = 0; j < N_node; j++ )
		{
			matrix_row<matrix<double> > instance_j(data,j);
// center around current instance
			instance_j -= instance;
// calculate the norm
			data_norm(j) = norm_2(instance_j);
		}
// sort-indices to find the instances within max_radius
		vector<double> sorted_indices = get_ranked_indices( data_norm );
		for( size_t k = 0; k < N_node; k++ )
		{
			double radius = data_norm(k);
			if( radius > max_radius )
			{
				break;
			}
			else
			{
				A(i,k) = 1;
			}
		}
	}
// calculate the degree matrix
	compressed_matrix<size_t> D( N_node, N_node );
	for( size_t i = 0; i < N_node; ++i)
	{
		matrix_row< compressed_matrix<size_t> > row_i(A,i);
		D(i,i) = std::count(row_i.begin(), row_i.end(), 1);
	}
// calculate the laplacian matrix
	compressed_matrix<size_t> L = D - A;
// split the data according to the second eigenvector of L
// first do Singular value decomposition
	vector<double> eigvals; // vector of EVs
// do eigenvalue decomposition
//TODO make eigenvalue stuff work
	//syev('V', L, eigvals);
// TODO
// REST IS DUMMY!
	size_t N_l = 0;
	size_t N_r = 0;
	std::cout << "Splitted " << N_node << " data points: Points to the left: " << N_l << " points to the right: " << N_r << std::endl;
	node_t* node_l(new node_t);
	//node_l->set_data(data_l);
	node_t* node_r(new node_t);
	//node_r->set_data(data_r);
	return std::array<node_t*, 2>{ {node_l, node_r} };  
}
