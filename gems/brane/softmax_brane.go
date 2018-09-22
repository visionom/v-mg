package brane

import (
	"math"

	"github.com/visionom/vision-mg/gems/mtx"
)

func softmax(a mtx.Mtx, c float64) mtx.Mtx {
	var sum float64
	data := make([]float64, len(a.Data))
	for j := 0; j < a.Shape[0]; j++ {
		sum = 0
		for i := 0; i < a.Shape[1]; i++ {
			v := a.Data[j*a.Shape[1]+i]
			sum += math.Exp(v - c)
		}
		for i := 0; i < a.Shape[1]; i++ {
			v := a.Data[j*a.Shape[1]+i]
			data[j*a.Shape[1]+i] = math.Exp(v-c) / sum
		}
	}
	a.Data = data
	return a
}
