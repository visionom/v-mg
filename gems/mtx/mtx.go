package mtx

import (
	"fmt"

	"github.com/visionom/vision-mg/gems/blas"
)

type Shape = []int

type Mtx struct {
	Shape Shape
	Data  []float64
}

func NewMtx(s Shape) Mtx {
	sum := 1
	for _, v := range s {
		sum *= v
	}
	return Mtx{
		s,
		make([]float64, sum),
	}
}

func Mul(a, b Mtx) Mtx {
	ra := Mtx{}
	if len(a.Shape) == 2 && len(b.Shape) == 2 && a.Shape[1] == b.Shape[0] {
		ra.Data = blas.Dgemm('N', 'N', a.Shape[0], b.Shape[1], a.Shape[1], 1, a.Data, 1, b.Data, 1, 0, nil, 1)
		ra.Shape = Shape{a.Shape[0], b.Shape[1]}
	} else {
		fmt.Println("error", a.Shape, b.Shape)
	}
	return ra
}

func MulAT(a, b Mtx) Mtx {
	if len(a.Shape) == 2 && len(b.Shape) == 2 {
		a.Data = blas.Dgemm('T', 'N', a.Shape[1], b.Shape[1], a.Shape[0], 1, a.Data, 1, b.Data, 1, 0, nil, 1)
		a.Shape[0] = a.Shape[1]
		a.Shape[1] = b.Shape[1]
	}
	return a
}

func Plus(a, b Mtx) Mtx {
	a.Data = blas.Daxpy(1.0, a.Data, b.Data)
	return a
}

func Axpy(a float64, x Mtx, y Mtx) Mtx {
	x.Data = blas.Daxpy(a, x.Data, y.Data)
	return x
}

func GetMtxCols(a Mtx, cols []int) Mtx {
	lcols := len(cols)
	s := Shape{a.Shape[0], lcols}
	ra := NewMtx(s)
	for j, col := range cols {
		for i := 0; i < a.Shape[0]; i++ {
			ra.Data[i*s[1]+j] = a.Data[i*a.Shape[1]+col]
		}
	}
	return ra
}

func GetMtxRows(a Mtx, rows []int) Mtx {
	lrows := len(rows)
	s := Shape{lrows, a.Shape[1]}
	ra := NewMtx(s)
	for j, k := range rows {
		for i := 0; i < s[1]; i++ {
			ra.Data[j*s[1]+i] = a.Data[k*s[0]+i]
		}
	}
	return ra
}
