package d

import (
	"fmt"

	"github.com/visionom/vision-mg/gems/mtx"
)

func NumericalGradient(loss func(mtx.Mtx) float64, a mtx.Mtx, h float64) mtx.Mtx {
	result := mtx.NewMtx(a.Shape)
	for i, k := range a.Data {
		a.Data[i] = k + h
		dx1 := loss(a)
		a.Data[i] = k - h
		dx2 := loss(a)
		a.Data[i] = k
		result.Data[i] = (dx1 - dx2) / (2 * h)
	}
	return result
}

func GradientDescent(loss func(mtx.Mtx) float64, init_x mtx.Mtx, stepNum int, rate float64) mtx.Mtx {
	x := init_x
	for i := 0; i < stepNum; i++ {
		r := loss(x)
		fmt.Println(r)
		grad := NumericalGradient(loss, x, rate)
		x = mtx.Axpy(-1*rate, grad, x)
	}
	return x
}
