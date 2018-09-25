package brane

import (
	"math"

	"github.com/visionom/vision-mg/gems/mtx"
)

func sigmoid(a mtx.Mtx) mtx.Mtx {
	for i, v := range a.GetData() {
		a.VSet(i, 1.0/float64(1.0+math.Exp(-1.0*v)))
	}
	return a
}
