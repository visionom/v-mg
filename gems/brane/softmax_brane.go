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
	ra := mtx.NewMtx(x.Shape)
	for i := 0; i < ra.Shape[0]; i++ {
		r := x.GetRow(i)
		sum := 0.0
		for j, x := range r.GetData() {
			ra.Set(i, j, math.Exp(x+1.0e-7))
			sum += ra.Get(i, j)
		}
		if sum == 0.0 {
			s[i] = 1.0
			for j := range r.GetData() {
				ra.Set(i, j, 0.1)
			}
		} else {
			s[i] = sum
			for j := range r.GetData() {
				ra.Set(i, j, ra.Get(i, j)/sum)
			}
		}
	}
	brn.s = s
	return ra
}

func (brn *SoftmaxBrane) Backward(dout mtx.Mtx) mtx.Mtx {
	dx := mtx.NewMtx(dout.Shape)
	for i, idout := range dout.GetData() {
		dx.VSet(i, (brn.s[i%dout.Shape[0]]-math.Log(idout+1.0e-7))/brn.s[i%dout.Shape[0]])
	}
	return dx
}
