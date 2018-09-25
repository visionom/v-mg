package mtx

import (
	"fmt"

	"github.com/visionom/vision-mg/gems/blas"
)

func Mul(a, b Mtx) Mtx {
	ra := Mtx{}
	if len(a.Shape) == 2 && len(b.Shape) == 2 && a.Shape[1] == b.Shape[0] {
		ra.SetData(blas.Dgemm('N', 'N', a.Shape[0], b.Shape[1], a.Shape[1], 1, a.GetData(), 1, b.GetData(), 1, 0, nil, 1))
		ra.Shape = Shape{a.Shape[0], b.Shape[1]}
	} else {
		fmt.Println("error", a.Shape, b.Shape)
	}
	return ra
}

func MulAT(a, b Mtx) Mtx {
	if len(a.Shape) == 2 && len(b.Shape) == 2 {
		a.SetData(blas.Dgemm('T', 'N', a.Shape[1], b.Shape[1], a.Shape[0], 1, a.GetData(), 1, b.GetData(), 1, 0, nil, 1))
		a.Shape[0] = a.Shape[1]
		a.Shape[1] = b.Shape[1]
	}
	return a
}
