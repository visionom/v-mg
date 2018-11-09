package brane

type Brane interface {
	Forward()
	Backward()
}

var ReduceMean = reduceMean

var Normpdf = normpdf

var MaxIndex = maxIndex

var MeanSquaredErr = meanSquaredErr
