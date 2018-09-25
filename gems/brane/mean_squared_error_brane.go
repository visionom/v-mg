package brane

import (
	"math"

	"github.com/visionom/vision-mg/gems/mtx"
)

func meanSquaredErr(a mtx.Mtx, b mtx.Mtx) mtx.Mtx {
	m := mtx.NewMtx(mtx.Shape{a.Shape[0], 1})
	for j := 0; j < a.Shape[0]; j++ {
		vs := a.GetRow(j)
		for i, v := range vs.GetData() {
			m.VSet(j, m.VGet(j)+math.Pow(v-b.Get(i, j), 2))
		}
	}

	return mtx.Ax(0.5, m)
}
