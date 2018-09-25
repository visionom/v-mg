package brane

import (
	"github.com/visionom/vision-mg/gems/mtx"
)

func maxIndex(a mtx.Mtx) []int {
	indexes := make([]int, a.Shape[0])
	for j := 0; j < a.Shape[0]; j++ {
		index := 0
		vs := a.GetRow(j)
		for i, v := range vs.GetData() {
			if v > a.Get(j, index) {
				index = i
			}
		}
		indexes[j] = index
	}
	return indexes
}
