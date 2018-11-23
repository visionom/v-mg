#! /bin/bash

dir="./MNIST_data"
mkdir -p $dir
cd $dir

files="train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz"

for file in $files
do
    rm -rf $file
    wget http://yann.lecun.com/exdb/mnist/$file
    gzip -d $file
done
