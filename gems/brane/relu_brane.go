package brane

import (
	"github.com/visionom/vision-mg/gems/mtx"
)

func relu(a mtx.Mtx) mtx.Mtx {
	for i, v := range a.Data {
		if v < 0 {
			a.Data[i] = 0
		}
	}
	return a
}
