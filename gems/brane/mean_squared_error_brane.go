package brane

import (
	"math"

	"github.com/visionom/vision-mg/gems/mtx"
)

func meanSquaredErr(a mtx.Mtx, b mtx.Mtx) mtx.Mtx {
	m := mtx.NewMtx(mtx.Shape{a.Shape[0], 1})
	for j := 0; j < a.Shape[0]; j++ {
		for i, v := range a.Data[j*a.Shape[1] : (j+1)*a.Shape[1]] {
			m.Data[j] += math.Pow(v-b.Data[j*a.Shape[1]+i], 2)
		}
	}

	for i := range m.Data {
		m.Data[i] *= 0.5
	}
	return m
}
