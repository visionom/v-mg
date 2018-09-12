package blas

func Daxpy(a float64, x []float64, y []float64) []float64 {
	for i, xi := range x {
		y[i] = a*xi + y[i]
	}
	return y
}

func Dgemm(transA, transB rune, m, n, k int, alpha float64, a []float64, lda int, b []float64, ldb int, beta float64, c []float64, ldc int) (rc []float64) {
	rc = make([]float64, m*n)
	noTA := false
	noTB := false

	if transA == 'N' || transA == 'n' {
		noTA = true
	}

	if transB == 'N' || transB == 'n' {
		noTB = true
	}

	if alpha == 0 {
		if beta == 0 {
			for j := 0; j < n; j++ {
				for i := 0; i < m; i++ {
					rc[j*m+i] = 0
				}
			}
			return
		}
		for j := 0; j < n; j++ {
			for i := 0; i < m; i++ {
				rc[j*m+i] = beta * c[j*m+i]
			}
		}
		return
	}

	if noTB {
		if noTA {
			// C := alpha*A*B + beta*C.
			for j := 0; j < n; j++ {
				if beta != 0 {
					for i := 0; i < m; i++ {
						rc[i*n+j] = beta * c[i*n+j]
					}
				}
				for l := 0; l < k; l++ {
					t := alpha * b[l*n+j]
					for i := 0; i < m; i++ {
						rc[i*n+j] += t * a[i*k+l]
					}
				}
			}
		} else {
			// C := alpha*A**T*B + beta*C
			for j := 0; j < n; j++ {
				for i := 0; i < m; i++ {
					var t float64
					for l := 0; l < k; l++ {
						t += a[l*m+i] * b[l*n+j]
					}
					if beta == 0 {
						rc[i*n+j] = alpha * t
					} else {
						rc[i*n+j] = alpha*t + beta*c[i*n+j]
					}
				}
			}
		}
	} else {
		if noTA {
			// C := alpha*A*B**T + beta*C
			for j := 0; j < n; j++ {
				if beta != 0 {
					for i := 0; i < m; i++ {
						rc[i*n+j] = beta * c[i*n+j]
					}
				}
				for l := 0; l < k; l++ {
					t := alpha * b[j*k+l]
					for i := 0; i < m; i++ {
						rc[i*n+j] += t * a[i*k+l]
					}
				}
			}
		} else {
			for j := 0; j < n; j++ {
				for i := 0; i < m; i++ {
					var t float64
					for l := 0; l < k; l++ {
						t += a[l*m+i] * b[j*k+l]
					}
					if beta == 0 {
						rc[i*n+j] = alpha * t
					} else {
						rc[i*n+j] = alpha*t + beta*c[i*n+j]
					}
				}
			}
			// C := alpha*A**T*B**T + beta*C
		}

	}
	return
}

func Dgemv() []float64 {
	return nil
}
