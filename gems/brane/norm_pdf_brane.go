package brane

import (
	"math/rand"
	"time"

	"github.com/visionom/vision-mg/gems/mtx"
)

func normpdf(a mtx.Mtx, m, d float64) mtx.Mtx {
	r1 := rand.New(rand.NewSource(time.Now().UnixNano()))
	for i := range a.GetData() {
		a.VSet(i, (r1.NormFloat64()*d + m))
	}
	return a
}
