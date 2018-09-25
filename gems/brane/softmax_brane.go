package brane

import (
	"math"

	"github.com/visionom/vision-mg/gems/mtx"
)

func softmax(a mtx.Mtx, c float64) mtx.Mtx {
	var sum float64
	ra := mtx.NewMtx(a.Shape)
	for j := 0; j < a.Shape[0]; j++ {
		sum = 0
		for i := 0; i < a.Shape[1]; i++ {
			v := a.Get(i, j)
			sum += math.Exp(v - c)
		}
		for i := 0; i < a.Shape[1]; i++ {
			v := a.Get(i, j)
			ra.Set(i, j, math.Exp(v-c)/sum)
		}
	}
	return a
}
