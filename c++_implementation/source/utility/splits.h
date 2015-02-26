#pragma once

#include "node_t.h"

std::array<node_t*, 2> split_node_default(
    node_t * node, const bool dim_shuffle, const size_t num_shuffle, const bool record);

std::array<node_t*, 2> split_node_gradient(
    node_t * node, const size_t nearest_neighbors, const bool record);

std::array<node_t*, 2> split_node_graph(
    node_t * node, const size_t max_radius, const bool record );

