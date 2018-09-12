package blas

func Daxpy(a float64, x []float64, y []float64) []float64 {
	for i, xi := range x {
		y[i] = a*xi + y[i]
	}
	return y
}
