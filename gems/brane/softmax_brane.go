package brane

import (
	"math"

	"github.com/visionom/vision-mg/gems/mtx"
)

type SoftmaxBrane struct {
	s []float64
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

func softmax(a mtx.Mtx, c float64) mtx.Mtx {
	var sum float64
	ra := mtx.NewMtx(a.Shape)
	for j := 0; j < a.Shape[0]; j++ {
		sum = 0
		for i := 0; i < a.Shape[1]; i++ {
			v := a.Get(j, i)
			sum += math.Exp(v - c)
		}
		for i := 0; i < a.Shape[1]; i++ {
			v := a.Get(j, i)
			ra.Set(j, i, math.Exp(v-c)/sum)
		}
	}
	return ra
}
