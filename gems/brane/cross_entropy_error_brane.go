package brane

import (
	"math"

	"github.com/visionom/vision-mg/gems/mtx"
)

type CrossEntropyErrorBrane struct {
	t mtx.Mtx
	x mtx.Mtx
}

func NewCrossEntropyErrorBrane(t mtx.Mtx) CrossEntropyErrorBrane {
	return CrossEntropyErrorBrane{t: t}
}

func (brn *CrossEntropyErrorBrane) Forward(x mtx.Mtx) mtx.Mtx {
	var e float64
	brn.x = x.Clone()
	n := brn.t.Shape[0]
	for i, ti := range brn.t.GetData() {
		e -= ti * math.Log(x.VGet(i)+1e-7)
	}
	return mtx.NewMtxNE(mtx.Shape{1, 1}, []float64{(e + 1e-7) / float64(n)})
}

func (brn *CrossEntropyErrorBrane) Backward(dout mtx.Mtx) mtx.Mtx {
	dx := mtx.NewMtx(brn.x.Shape)
	for i, ix := range brn.x.GetData() {
		dx.VSet(i, -1*brn.t.VGet(i)/(ix+1e-7))
	}
	return dx
}
