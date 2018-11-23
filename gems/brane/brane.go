package brane

import (
	"github.com/visionom/vision-mg/gems/mtx"
)

type Brane interface {
	Forward(x mtx.Mtx) mtx.Mtx
	Backward(dout mtx.Mtx) mtx.Mtx
}

type Branes struct {
	InitBranes func([]mtx.Mtx, bool, mtx.Mtx) []Brane
	GetGrad    func([]Brane) []mtx.Mtx
}

func (b *Branes) Forward(input mtx.Mtx, branes []Brane) mtx.Mtx {
	x := input.Clone()
	for _, brane := range branes {
		x = brane.Forward(x.Clone())
	}
	return x
}

func (bs *Branes) Backward(dout mtx.Mtx, branes []Brane) {
	for i := len(branes); i > 0; i-- {
		dout = branes[i-1].Backward(dout.Clone())
	}
}

func (bs *Branes) Predict(input mtx.Mtx, ms []mtx.Mtx) mtx.Mtx {
	branes := bs.InitBranes(ms, false, mtx.Mtx{})
	o := bs.Forward(input.Clone(), branes)
	return o
}

var ReduceMean = reduceMean

var Normpdf = normpdf

var MaxIndex = maxIndex

var MeanSquaredErr = meanSquaredErr
