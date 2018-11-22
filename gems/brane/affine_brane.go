package brane

import (
	"github.com/visionom/vision-mg/gems/mtx"
)

type AffineBrane struct {
	x  mtx.Mtx
	W  mtx.Mtx
	Dw mtx.Mtx
	B  float64
	Db float64
}

func NewAffineBrane(w mtx.Mtx, b float64) AffineBrane {
	return AffineBrane{
		W: w,
		B: b,
	}
}

func (brn *AffineBrane) Forward(x mtx.Mtx) mtx.Mtx {
	brn.x = x.Clone()
	return mtx.MulBeta(x.Clone(), brn.W.Clone(), brn.B)
}

func (brn *AffineBrane) Backward(dout mtx.Mtx) mtx.Mtx {
	dx := mtx.MulBT(dout.Clone(), brn.W.Clone())
	brn.Dw = mtx.MulAT(brn.x.Clone(), dout.Clone())
	brn.Db = mtx.Sum(dout.Clone())
	return dx
}
