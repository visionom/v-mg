package main

import (
	"fmt"
	"log"
	"math/rand"
	"os"
	"os/exec"
	"time"

	"github.com/visionom/vision-mg/gems/brane"
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
	TrainData := readImage("./MNIST_data/train-images-idx3-ubyte")
	TrainLabel := readLabel("./MNIST_data/train-labels-idx1-ubyte")

	//TrainData := readImage("./MNIST_data/t10k-images-idx3-ubyte")
	//TrainLabel := readLabel("./MNIST_data/t10k-labels-idx1-ubyte")

	input := TrainData
	label := TrainLabel
	//ms := readM()

	dataSize := 784
	n1Size := 16
	oSize := 10
	bSize := 2

	ms := make([]mtx.Mtx, 3)
	ms[0] = mtx.NewMtx(mtx.Shape{dataSize, n1Size})
	ms[1] = mtx.NewMtx(mtx.Shape{n1Size, oSize})
	//ms[0] = brane.Normpdf(mtx.NewMtx(mtx.Shape{dataSize, n1Size}), 1, 1)
	//ms[1] = brane.Normpdf(mtx.NewMtx(mtx.Shape{n1Size, n2Size}), 1, 1)
	//ms[2] = brane.Normpdf(mtx.NewMtx(mtx.Shape{n2Size, oSize}), 1, 1)
	ms[2] = mtx.NewMtx(mtx.Shape{1, bSize})
	ms[2].VSet(0, 3)
	ms[2].VSet(1, 1)

	o := predict(input.GetRow(1), ms[0], ms[1], ms[2])
	log.Printf("%+.7v", input.GetRow(1))
	log.Println(o)
	log.Println(label[1])

	ms = BGradientDescent(input, label, ms, 1000, 0.01)
}

func getRandIntList(size, n int) []int {
	l := make([]int, n)
	r1 := rand.New(rand.NewSource(time.Now().UnixNano()))
	for i := 0; i < n; i++ {
		l[i] = r1.Int() % size
	}
	return l
}

func get(m mtx.Mtx, l []int, nums int) (mtx.Mtx, mtx.Mtx) {
	p := getRandIntList(m.Shape[0], nums)
	label := make([]int, nums)
	for j, k := range p {
		label[j] = l[k]
	}
	input := m.GetRows(p)
	ml := initT(mtx.Shape{nums, 10}, label)
	return input, ml
}

func initT(s mtx.Shape, label []int) mtx.Mtx {
	t := mtx.NewMtx(s)
	for i, v := range label {
		t.Set(v, i, 1.0)
	}
	return t
}

func model(input, w1, wo, bs mtx.Mtx) mtx.Mtx {
	b1 := brane.NewSigmoidBrane()
	h1 := b1.Backward(mtx.Aopy(bs.VGet(0), mtx.Mul(input, w1)))

	o := mtx.Aopy(bs.VGet(1), mtx.Mul(h1, wo))
	return o
}

func predict(input, w1, wo, bs mtx.Mtx) []int {
	o := model(input, w1, wo, bs)
	return brane.MaxIndex(o)
}

func accuracy(input, w1, wo, bs, t mtx.Mtx) float64 {
	o := model(input, w1, wo, bs)
	b := brane.NewSoftmaxCrossEntropyLossBrane()
	errs := b.Forward(o, t)
	return errs
}

func lossf(input, t mtx.Mtx, p []mtx.Mtx, n int) func(mtx.Mtx) float64 {
	return func(x mtx.Mtx) float64 {
		p[n] = x
		return accuracy(input, p[0].Clone(), p[1].Clone(), p[2].Clone(), t)
	}
}

func logMtx(s string, m mtx.Mtx) {
	//log.Printf(">%s\n%v\n", s, m)
}

func NumericalGradient(loss func(mtx.Mtx) float64, a mtx.Mtx, h float64) mtx.Mtx {
	result := mtx.NewMtx(a.Shape)
	s := float64(len(a.GetData()))
	for i, k := range a.GetData() {
		a.VSet(i, k+h)
		dx1 := loss(a)
		a.VSet(i, k-h)
		dx2 := loss(a)
		a.VSet(i, k)
		result.VSet(i, (dx1-dx2)/(2*h))
		fmt.Printf("%f\r", float64(i)/s)
	}
	return result
}

func BGradientDescent(input mtx.Mtx, t []int, ms []mtx.Mtx, stepNum int, rate float64) []mtx.Mtx {
	linput, lt := get(input, t, 10)
	for j := 0; j < stepNum; j++ {
		var r float64
		for i := 0; i < len(ms); i++ {
			f := lossf(linput, lt, ms, i)
			x := ms[i].Clone()
			r = f(x)
			grad := NumericalGradient(f, x, rate)
			x = mtx.Axpy(-1*rate, grad, x)
			ms[i] = x
			//nr := f(x)
		}
		saveM(ms)
		cmd := exec.Command("clear") //Linux example, its tested
		cmd.Stdout = os.Stdout
		cmd.Run()
		log.Println(r)
		o := predict(linput, ms[0], ms[1], ms[2])
		log.Println(o)
		log.Println(brane.MaxIndex(lt))
	}
	return ms
}
