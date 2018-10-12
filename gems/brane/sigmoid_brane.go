package brane

import (
	"math"

	"github.com/visionom/vision-mg/gems/mtx"
)

type SigmoidBrane struct{}

func NewSigmoid() SigmoidBrane {
	return SigmoidBrane{}
}

func (brn *SigmoidBrane) Forward(x mtx.Mtx) mtx.Mtx {
	rx := mtx.NewMtx(x.Shape)
	for i, v := range x.GetData() {
		rx.VSet(i, 1.0/float64(1.0+math.Exp(-1.0*v)))
	}
	return rx
}

func (brn *SigmoidBrane) Backward(dout mtx.Mtx) mtx.Mtx {
	dx := mtx.NewMtx(dout.Shape)
	for i, v := range dout.GetData() {
		dx.VSet(i, v*(1-v))
	}
	return dx
}

func sigmoid(a mtx.Mtx) mtx.Mtx {
	ra := mtx.NewMtx(a.Shape)
	for i, v := range a.GetData() {
		ra.VSet(i, 1.0/float64(1.0+math.Exp(-1.0*v)))
	}
	return ra
}
