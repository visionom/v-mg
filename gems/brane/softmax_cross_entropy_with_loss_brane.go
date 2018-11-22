package brane

import (
	"github.com/visionom/vision-mg/gems/mtx"
)

type SoftmaxCrossEntropyLossBrane struct {
	loss mtx.Mtx
	y    mtx.Mtx
	t    mtx.Mtx
	b1   SoftmaxBrane
	b2   CrossEntropyErrorBrane
}

func NewSoftmaxCrossEntropyLossBrane(t mtx.Mtx) SoftmaxCrossEntropyLossBrane {
	b1 := NewSoftmaxBrane()
	b2 := NewCrossEntropyErrorBrane(t)
	return SoftmaxCrossEntropyLossBrane{t: t, b1: b1, b2: b2}
}

func (brn *SoftmaxCrossEntropyLossBrane) Forward(x mtx.Mtx) mtx.Mtx {
	brn.y = brn.b1.Forward(x)
	//fmt.Println(brn.y.GetData())
	brn.loss = brn.b2.Forward(brn.y)
	//fmt.Println(brn.loss.GetData())
	return brn.loss
}

func (brn *SoftmaxCrossEntropyLossBrane) Backward(dout mtx.Mtx) mtx.Mtx {
	dx := mtx.Ax(1.0/float64(brn.t.Shape[0]), mtx.Axpy(-1, brn.t, brn.y))
	return dx
}
