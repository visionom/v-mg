package brane

import "github.com/visionom/vision-mg/gems/mtx"

type AffineBrane struct {
	x  mtx.Mtx
	w  mtx.Mtx
	dw mtx.Mtx
	b  float64
	db float64
}

func NewAffineBrane(w mtx.Mtx, b float64) AffineBrane {
	return AffineBrane{
		w: w,
		b: b,
	}
}

func (brn *AffineBrane) Forward(x mtx.Mtx) mtx.Mtx {
	brn.x = x
	return mtx.Mul(x, brn.w)
}

func (brn *AffineBrane) Backward(dout mtx.Mtx) mtx.Mtx {
	dx := mtx.MulBT(dout, brn.w)
	brn.dw = mtx.MulAT(brn.x, dout)
	brn.db = mtx.Sum(dout)
	return dx
}
