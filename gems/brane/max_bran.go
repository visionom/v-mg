package brane

import (
	"github.com/visionom/vision-mg/gems/mtx"
)

func maxIndex(a mtx.Mtx) []int {
	indexes := make([]int, a.Shape[0])
	for j := 0; j < a.Shape[0]; j++ {
		index := 0
		for i, v := range a.Data[j*a.Shape[1] : (j+1)*a.Shape[1]] {
			if v > a.Data[index+j*a.Shape[1]] {
				index = i
			}
		}
		indexes[j] = index
	}
	return indexes
}
