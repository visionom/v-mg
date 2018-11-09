package brane

import (
	"github.com/visionom/vision-mg/gems/mtx"
)

type ReluBrane struct {
	mask mtx.Mtx
}

func NewReluBrane() ReluBrane {
	return ReluBrane{}
}

func (brn *ReluBrane) Forward(x mtx.Mtx) mtx.Mtx {
	brn.mask = mtx.NewMtx(x.Shape)
	out := mtx.NewMtx(x.Shape)
	for i, v := range x.GetData() {
		if v < 0 {
			brn.mask.VSet(i, 0)
		} else {
			out.VSet(i, v)
			brn.mask.VSet(i, 1)
		}
	}
	return out
}

func (brn *ReluBrane) Backward(dout mtx.Mtx) mtx.Mtx {
	dx := mtx.NewMtx(dout.Shape)
	for i, v := range brn.mask.GetData() {
		dx.VSet(i, v*dout.VGet(i))
	}
	return dx
}
