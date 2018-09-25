package main

import (
	"fmt"
	"log"
	"math/rand"
	"time"

	"github.com/visionom/vision-mg/gems/brane"
	"github.com/visionom/vision-mg/gems/d"
	"github.com/visionom/vision-mg/gems/mtx"
)

type session struct {
	input mtx.Mtx
	label []int
	w1    mtx.Mtx
	w2    mtx.Mtx
	wo    mtx.Mtx
	bs    mtx.Mtx
	t     mtx.Mtx
}

func main() {
	TrainData := readImage("./MNIST_data/train-images-idx3-ubyte.gz")
	TrainLabel := readLabel("./MNIST_data/train-labels-idx1-ubyte.gz")

	//TestData := readImage("./MNIST_data/t10k-images-idx3-ubyte.gz")
	//Testlabel := readLabel("./MNIST_data/t10k-labels-idx1-ubyte.gz")

	nums := 1000
	dataSize := 784
	n1Size := 1024
	n2Size := 625
	oSize := 10
	bSize := 3

	s := session{}
	{
		s.input, s.label = get(TrainData, TrainLabel, nums)
		s.w1 = brane.Normpdf(mtx.NewMtx(mtx.Shape{dataSize, n1Size}), 0.001, 3)
		s.w2 = brane.Normpdf(mtx.NewMtx(mtx.Shape{n1Size, n2Size}), 0.001, 3)
		s.wo = brane.Normpdf(mtx.NewMtx(mtx.Shape{n2Size, oSize}), 0.001, 3)
		s.bs = mtx.NewMtx(mtx.Shape{1, bSize})
		s.t = initT(mtx.NewMtx(mtx.Shape{oSize, nums}), s.label)
	}
	o := predict(s.input, s.w1, s.w2, s.wo, s.bs)
	fmt.Println(o)
	fmt.Println(s.label)

	s.w1 = d.GradientDescent(lossf(s), s.w1, 1000, 10)

	o = predict(s.input, s.w1, s.w2, s.wo, s.bs)
	fmt.Println(o)
	fmt.Println(s.label)

}

func getRandIntList(size, n int) []int {
	l := make([]int, n)
	r1 := rand.New(rand.NewSource(time.Now().UnixNano()))
	for i := 0; i < n; i++ {
		l[i] = r1.Int() % size
	}
	return l
}

func get(m mtx.Mtx, l []int, nums int) (mtx.Mtx, []int) {
	p := getRandIntList(m.Shape[0], nums)
	label := make([]int, nums)
	for j, k := range p {
		label[j] = l[k]
	}
	input := m.GetRows(p)
	return input, label
}

func initT(t mtx.Mtx, label []int) mtx.Mtx {
	for i, v := range label {
		t.Set(v, i, 1.0)
	}
	return t
}

func model(input, w1, w2, wo, bs mtx.Mtx) mtx.Mtx {
	logMtx("input", input)
	logMtx("w1", w1)
	h1 := brane.Sigmoid(mtx.Mul(input, w1))
	logMtx("h1", h1)

	logMtx("wo", wo)
	h2 := brane.Sigmoid(mtx.Mul(h1, w2))
	logMtx("h2", h2)

	logMtx("wo", wo)
	o := mtx.Mul(h2, wo)
	logMtx("o", o)
	return o
}

func predict(input, w1, w2, wo, bs mtx.Mtx) []int {
	o := model(input, w1, w2, wo, bs)
	return brane.MaxIndex(o)
}

func accuracy(input, w1, w2, wo, bs, t mtx.Mtx) float64 {
	o := model(input, w1, w2, wo, bs)
	so := brane.Softmax(o, 0)
	errs := brane.MeanSquaredErr(so, t)
	return brane.ReduceMean(errs)
}

func lossf(s session) func(mtx.Mtx) float64 {
	return func(x mtx.Mtx) float64 {
		return accuracy(s.input, x, s.w2, s.wo, s.bs, s.t)
	}
}

func logMtx(s string, m mtx.Mtx) {
	a := ">>>>>>>>>"
	log.SetPrefix(s + " " + a[0:len(a)-len(s)] + " ")
	log.Println(m.Shape)
	//for i := 0; i < m.Shape[0]; i++ {
	//	log.Println(m.Data[i*m.Shape[1] : (i+1)*m.Shape[1]])
	//}
	log.SetPrefix("")
}
