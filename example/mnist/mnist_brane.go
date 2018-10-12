package main

import (
	"log"
	"math/rand"
	"time"

	"github.com/visionom/vision-mg/gems/brane"
	"github.com/visionom/vision-mg/gems/mtx"
)

func main() {
	//TrainData := readImage("./MNIST_data/train-images-idx3-ubyte.gz")
	//TrainLabel := readLabel("./MNIST_data/train-labels-idx1-ubyte.gz")

	TrainData := readImage("./MNIST_data/t10k-images-idx3-ubyte")
	TrainLabel := readLabel("./MNIST_data/t10k-labels-idx1-ubyte")

	input := TrainData
	label := TrainLabel
	//ms := readM()

	dataSize := 784
	n1Size := 1024
	n2Size := 625
	oSize := 10
	bSize := 3

	ms := make([]mtx.Mtx, 4)
	ms[0] = brane.Normpdf(mtx.NewMtx(mtx.Shape{dataSize, n1Size}), 2, 1)
	ms[1] = brane.Normpdf(mtx.NewMtx(mtx.Shape{n1Size, n2Size}), 5, 1)
	ms[2] = brane.Normpdf(mtx.NewMtx(mtx.Shape{n2Size, oSize}), 1, 1)
	ms[3] = mtx.NewMtx(mtx.Shape{1, bSize})
	ms[3].VSet(0, 1)
	ms[3].VSet(1, 1)
	ms[3].VSet(2, 1)

	linput, lt := get(input, label, 3)

	bs := NewBranes(ms)
	o := bs.Predict(linput)
	l := bs.Loss(linput, initT(mtx.Shape{10, 3}, lt))
	//log.Printf("%+.28v\n", linput)
	log.Printf("%+v", o)
	log.Println(brane.MaxIndex(o))
	log.Println(lt)
	log.Println(l)

	ms = GradientDescent(input, label, ms, 1000, 0.1)
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

type Branes struct {
	base    []mtx.Mtx
	brn0    brane.AffineBrane
	brn1    brane.SigmoidBrane
	brn2    brane.AffineBrane
	brn3    brane.SigmoidBrane
	brn4    brane.AffineBrane
	brnLast brane.SoftmaxCrossEntropyLossBrane
}

func NewBranes(ms []mtx.Mtx) Branes {
	return Branes{
		base:    ms,
		brn0:    brane.NewAffineBrane(ms[0], ms[3].VGet(0)),
		brn1:    brane.NewSigmoid(),
		brn2:    brane.NewAffineBrane(ms[1], ms[3].VGet(1)),
		brn3:    brane.NewSigmoid(),
		brn4:    brane.NewAffineBrane(ms[2], ms[3].VGet(2)),
		brnLast: brane.NewSoftmaxCrossEntropyLossBrane(),
	}
}

func (bs *Branes) Predict(x mtx.Mtx) mtx.Mtx {
	//log.Printf(">x:\n%+v", x)
	o1 := bs.brn0.Forward(x)
	//log.Printf(">o1:\n%+v", o1)
	o2 := bs.brn1.Forward(o1)
	//log.Printf(">o2:\n%+v", o2)
	o3 := bs.brn2.Forward(o2)
	//log.Printf(">o3:\n%+v", o3)
	o4 := bs.brn3.Forward(o3)
	//log.Printf(">o4:\n%+v", o4)
	o := bs.brn4.Forward(o4)
	//log.Printf(">o:\n%+v", o)
	return o
}

func (bs *Branes) Reasoning() mtx.Mtx {
	dout := bs.brnLast.Backward()
	dx4 := bs.brn4.Backward(dout)
	dx3 := bs.brn3.Backward(dx4)
	dx2 := bs.brn2.Backward(dx3)
	dx1 := bs.brn1.Backward(dx2)
	dx0 := bs.brn0.Backward(dx1)
	return dx0
}

func (bs *Branes) Loss(x mtx.Mtx, t mtx.Mtx) float64 {
	o := bs.Predict(x)
	y := bs.brnLast.Forward(o, t)
	return y
}

func (bs *Branes) GetGrads() []mtx.Mtx {
	gs := make([]mtx.Mtx, 4)
	gs[0] = bs.brn0.Dw
	gs[1] = bs.brn2.Dw
	gs[2] = bs.brn4.Dw
	gs[3] = mtx.NewMtx(bs.base[3].Shape)
	gs[3].SetData([]float64{
		bs.brn0.Db,
		bs.brn2.Db,
		bs.brn4.Db,
	})
	return gs
}

func getRandIntList(size, n int) []int {
	l := make([]int, n)
	r1 := rand.New(rand.NewSource(time.Now().UnixNano()))
	for i := 0; i < n; i++ {
		l[i] = r1.Int() % size
	}
	return l
}

func initT(s mtx.Shape, label []int) mtx.Mtx {
	t := mtx.NewMtx(s)
	for i, v := range label {
		t.Set(i, v, 1.0)
	}
	return t
}

func model(input, w1, w2, wo, bs mtx.Mtx) mtx.Mtx {
	h1 := brane.Sigmoid(mtx.Aopy(bs.VGet(0), mtx.Mul(input, w1)))

	h2 := brane.Sigmoid(mtx.Aopy(bs.VGet(1), mtx.Mul(h1, w2)))

	o := mtx.Aopy(bs.VGet(2), mtx.Mul(h2, wo))
	return o
}

func accuracy(input, w1, w2, wo, bs, t mtx.Mtx) float64 {
	o := model(input, w1, w2, wo, bs)
	so := brane.Softmax(o, 0)
	errs := brane.MeanSquaredErr(so, t)
	return brane.ReduceMean(errs)
}

func GradientDescent(input mtx.Mtx, t []int, ms []mtx.Mtx, stepNum int, rate float64) []mtx.Mtx {
	linput, lt := get(input, t, 10)
	mt := initT(mtx.Shape{10, 10}, lt)
	log.Println(lt)
	for i := 0; i < stepNum; i++ {
		bs := NewBranes(ms)
		r := bs.Loss(linput, mt)
		log.Println(r)
		bs.Reasoning()
		grads := bs.GetGrads()
		for i, g := range grads {
			ms[i] = mtx.Axpy(-1*rate, g, ms[i])
		}
		saveM(ms)
		o := bs.Predict(linput)
		//log.Printf("%+v", o)
		log.Println(brane.MaxIndex(o))
	}
	return ms
}
