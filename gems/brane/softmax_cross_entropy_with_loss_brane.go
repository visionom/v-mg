package brane

import (
	"github.com/visionom/vision-mg/gems/mtx"
)

type SoftmaxCrossEntropyLossBrane struct {
	loss float64
	y    mtx.Mtx
	t    mtx.Mtx
	b1   SoftmaxBrane
	b2   CrossEntropyErrorBrane
}

func NewSoftmaxCrossEntropyLossBrane() SoftmaxCrossEntropyLossBrane {
	return SoftmaxCrossEntropyLossBrane{}
}

func (brn *SoftmaxCrossEntropyLossBrane) Forward(x, t mtx.Mtx) float64 {
	brn.t = t
	brn.b1 = SoftmaxBrane{}
	brn.b2 = CrossEntropyErrorBrane{}
	brn.y = brn.b1.Forward(x)
	brn.loss = brn.b2.Forward(brn.t, brn.y)
	return brn.loss
}

func (brn *SoftmaxCrossEntropyLossBrane) Backward() mtx.Mtx {
	dx := mtx.Ax(1.0/float64(brn.t.Shape[0]), mtx.Axpy(-1, brn.t, brn.y))
	return dx
}
