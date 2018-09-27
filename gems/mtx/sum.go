package mtx

func Sum(a Mtx) float64 {
	var sum float64
	for _, v := range a.GetData() {
		sum += v
	}
	return sum
}
