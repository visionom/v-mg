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

func (brn *SoftmaxCrossEntropyLossBrane) Forward(x, t mtx.Mtx) float64 {
	brn.t = t
	brn.b1 = SoftmaxBrane{}
	brn.b2 = CrossEntropyErrorBrane{}
	brn.y = brn.b1.Forward(x)
	brn.loss = brn.b2.Forward(t, brn.y)
	return brn.loss
}

func (brn *SoftmaxCrossEntropyLossBrane) Backward() mtx.Mtx {
	dy := brn.b2.Backward()
	dx := brn.b1.Backward(dy)
	return dx
}
