package brane

import "github.com/visionom/vision-mg/gems/mtx"

func reduceMean(a mtx.Mtx) float64 {
	var sum float64
	for _, v := range a.Data {
		sum += v / float64(len(a.Data))
	}
	return sum
}
