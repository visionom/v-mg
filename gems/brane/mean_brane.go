package brane

import "github.com/visionom/vision-mg/gems/mtx"

func reduceMean(a mtx.Mtx) float64 {
	var sum float64
	lens := float64(len(a.GetData()))
	for _, v := range a.GetData() {
		sum += v / lens
	}
	return sum
}
