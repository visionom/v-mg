package brane

import (
	"math"

	"github.com/visionom/vision-mg/gems/mtx"
)

type CrossEntropyErrorBrane struct {
	t mtx.Mtx
	x mtx.Mtx
}

func (brn *CrossEntropyErrorBrane) Forward(t, x mtx.Mtx) float64 {
	var e float64
	for k, tk := range t.GetData() {
		e -= tk * math.Log(x.VGet(k))
	}
	return e
}

func (brn *CrossEntropyErrorBrane) Backward() mtx.Mtx {
	dx := mtx.NewMtx(brn.x.Shape)
	for i, ix := range brn.x.GetData() {
		dx.VSet(i, -1*brn.t.VGet(i)/ix)
	}
	return dx
}
