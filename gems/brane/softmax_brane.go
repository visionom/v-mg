package brane

import (
	"math"

	"github.com/visionom/vision-mg/gems/mtx"
)

type SoftmaxBrane struct {
	s []float64
}

func NewSoftmaxBrane() SoftmaxBrane {
	return SoftmaxBrane{}
}

func (brn *SoftmaxBrane) Forward(x mtx.Mtx) mtx.Mtx {
	s := make([]float64, x.Shape[0])
	e := mtx.NewMtx(x.Shape)
	var c float64
	rx := mtx.NewMtx(x.Shape)
	for j := 0; j < x.Shape[0]; j++ {
		for i := 0; i < x.Shape[1]; i++ {
			e.Set(j, i, math.Exp(x.Get(j, i)-c))
			s[j] += e.Get(j, i)
		}
		for i := 0; i < rx.Shape[1]; i++ {
			rx.Set(j, i, e.Get(j, i)/s[j])
		}
	}
	return rx
}

func (brn *SoftmaxBrane) Backward(dout mtx.Mtx) mtx.Mtx {
	dx := mtx.NewMtx(dout.Shape)
	for i, idout := range dout.GetData() {
		dx.VSet(i, (brn.s[i]-math.Log(idout))/brn.s[i])
	}
	return dx
}
