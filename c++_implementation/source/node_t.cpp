#include <algorithm>
#include <boost/numeric/ublas/matrix_proxy.hpp>

#include "node_t.h"

node_t::node_t() : 	mData(),
   					mChildren(),
					mTerminal(false),
					mProbability(0.),
					mVolume(0.),
				    mDepth(0),
					mSplit_dimension(0),
				   	mSplit_threshold(0.)
{}
							
void node_t::set_data(const image_data_t & data)
{
	mData = data;
	calculate_volume();
}
	
const image_data_t & node_t::get_data() const
{
	return mData;
}

void node_t::add_child(node_t* child)
{
	if( mChildren.size() < max_nodes )
	{
		mChildren.push_back(child);
	}
	else
	{
		throw std::runtime_error("node_t::add_child: Trying to add more than 2 nodes as children!");
	}
}
	
const node_t & node_t::get_child( const side_t side ) const
{
	return *(mChildren[side]);
}

node_t & node_t::get_child( const side_t side )
{
	return *(mChildren[side]);
}
	
void node_t::calculate_probability( const size_t N_class )
{
	size_t N_node = get_data().size1();
	double V_node = get_volume();
// calculate the probability for different cases
// 1st: Node data is empty or only one instance in node data -> set probability to 0
	if( (V_node == 0. && N_node == 0) || (V_node == 0. && N_node == 1) )
	{
		mProbability = 0.;
	}
// 2nd Node data is empty but node has non-vanishing volume -> this should not occur!!!
	else if( V_node != 0. && N_node == 0 )
	{
		throw std::runtime_error("node_t::calculate_probability: Node Volume is not zero for an empty node!");
	}
// 3rd: Node data has more than 1 instance
	else
	{
		mProbability = N_node / (V_node * N_class);
	}
}

double node_t::get_probability() const
{
	return mProbability;
}
	
void node_t::calculate_volume()
{
	using namespace boost::numeric::ublas;
	if( mData.size1() == 0 )
	{
		mVolume = 0.;
	}
	else
	{
		double vol = 1.;
		for( size_t d = 0; d < mData.size2(); d++ )
		{
			matrix_column<matrix<double> const> data_dim(mData, d);
			double dim_max = *( std::max_element( data_dim.begin(), data_dim.end() ) );
			double dim_min = *( std::min_element( data_dim.begin(), data_dim.end() ) );
			vol *= (dim_max - dim_min); 
		}
		mVolume = vol;
	}
}
	
double node_t::get_volume() const
{
	return mVolume;
}
	
void node_t::set_depth(const size_t depth)
{
	mDepth = depth;
}
	
size_t node_t::get_depth() const
{
	return mDepth;
}

void node_t::set_split_dimension(const size_t dim)
{
	mSplit_dimension = dim;
}

size_t node_t::get_split_dimension() const
{
	return mSplit_dimension;
}

void node_t::set_split_threshold(const double thresh)
{
	mSplit_threshold = thresh;
}

double node_t::get_split_threshold() const
{
	return mSplit_threshold;
}
	
void node_t::set_terminal(const bool terminal)
{
	mTerminal = terminal;
}
	
bool node_t::get_terminal() const
{
	return mTerminal;
}
