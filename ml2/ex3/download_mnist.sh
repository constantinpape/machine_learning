#!/bin/bash

mkdir -p datasets/mnist

if ! [ -e datasets/mnist/train-images-idx3-ubyte.gz ]
	then
		wget -P datasets/mnist/ http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
fi
gzip -d datasets/mnist/train-images-idx3-ubyte.gz

if ! [ -e datasets/mnist/train-labels-idx1-ubyte.gz ]
	then
		wget -P datasets/mnist/ http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
fi
gzip -d datasets/mnist/train-labels-idx1-ubyte.gz

if ! [ -e datasets/mnist/t10k-images-idx3-ubyte.gz ]
	then
		wget -P datasets/mnist/ http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
fi
gzip -d datasets/mnist/t10k-images-idx3-ubyte.gz

if ! [ -e datasets/mnist/t10k-labels-idx1-ubyte.gz ]
	then
		wget -P datasets/mnist/ http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
fi
gzip -d datasets/mnist/t10k-labels-idx1-ubyte.gz
