package brane

type Brane interface {
	Forward()
	Backward()
}

var ReduceMean = reduceMean

var Normpdf = normpdf

var Relu = relu

var Sigmoid = sigmoid

var Softmax = softmax

var MaxIndex = maxIndex

var MeanSquaredErr = meanSquaredErr
