package brane

import (
	"github.com/visionom/vision-mg/gems/mtx"
)

type AffineBrane struct {
	x  mtx.Mtx
	w  mtx.Mtx
	Dw mtx.Mtx
	b  float64
	Db float64
}

func NewAffineBrane(w mtx.Mtx, b float64) AffineBrane {
	return AffineBrane{
		w: w,
		b: b,
	}
}

func (brn *AffineBrane) Forward(x mtx.Mtx) mtx.Mtx {
	brn.x = x
	return mtx.MulBeta(x, brn.w, brn.b)
}

func (brn *AffineBrane) Backward(dout mtx.Mtx) mtx.Mtx {
	dx := mtx.MulBT(dout, brn.w)
	brn.Dw = mtx.MulAT(brn.x, dout)
	brn.Db = mtx.Sum(dout)
	return dx
}
