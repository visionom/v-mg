package brane

import (
	"github.com/visionom/vision-mg/gems/mtx"
)

type ReluBrane struct {
	mask mtx.Mtx
}

func (brn *ReluBrane) Forward(a mtx.Mtx) mtx.Mtx {
	brn.mask = mtx.NewMtx(a.Shape)
	out := mtx.NewMtx(a.Shape)
	for i, v := range a.Data {
		if v < 0 {
			out.Data[i] = 0
			brn.mask.Data[i] = 0
		} else {
			brn.mask.Data[i] = 1
		}
	}
	return out
}

func (brn *ReluBrane) Backward(dout mtx.Mtx) mtx.Mtx {
	dx := mtx.NewMtx(dout.Shape)
	for i, v := range brn.mask.Data {
		dx.Data[i] = v * dout.Data[i]
	}
	return dx
}

func relu(a mtx.Mtx) mtx.Mtx {
	for i, v := range a.Data {
		if v < 0 {
			a.Data[i] = 0
		}
	}
	return a
}
