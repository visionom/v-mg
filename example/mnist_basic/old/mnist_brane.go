package main

import (
	"fmt"
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

	ms := initMS()

	linput, lt := get(TrainData, TrainLabel, 1)
	bs := NewBranes(ms)
	o := bs.Predict(linput)

	log.Printf("result:\n%v", brane.MaxIndex(o))
	log.Printf("label:\n%v", lt)

	bs = NewBranes(ms)
	l := bs.Loss(linput, lt)
	bs.Reasoning()

	log.Printf("loss:\t%v", l)
	affirm(ms, linput, lt)

	ms = GradientDescent(input, label, ms, 1000, 0.1)
}

func initMS() []mtx.Mtx {
	//ms := readM()

	dataSize := 784
	n1Size := 1024
	n2Size := 625
	oSize := 10
	bSize := 3

	ms := make([]mtx.Mtx, 4)
	ms[0] = brane.Normpdf(mtx.NewMtx(mtx.Shape{dataSize, n1Size}), 0.1, 0.01)
	ms[1] = brane.Normpdf(mtx.NewMtx(mtx.Shape{n1Size, n2Size}), 0.1, 0.01)
	ms[2] = brane.Normpdf(mtx.NewMtx(mtx.Shape{n2Size, oSize}), 0.1, 0.01)
	ms[3] = mtx.NewMtx(mtx.Shape{1, bSize})
	ms[3].VSet(0, -0.5)
	ms[3].VSet(1, -50.0)
	ms[3].VSet(2, 0)

	//saveM(ms)
	return ms
}

func affirm(ms []mtx.Mtx, input, t mtx.Mtx) {
	h := 0.1
	for i := 0; i < len(ms); i++ {
		grad := mtx.NewMtx(ms[i].Shape)
		for j, k := range ms[i].GetData() {
			ms[i].VSet(j, k+h)
			bs := NewBranes(ms)
			dx1 := bs.Loss(input, t)
			ms[i].VSet(j, k-h)
			bs = NewBranes(ms)
			dx2 := bs.Loss(input, t)
			ms[i].VSet(j, k)
			dx := (dx1 - dx2) / (2 * h)
			fmt.Printf("\r%f", dx)
			if dx != 0 {
				log.Println(dx)
			}
			grad.VSet(j, (dx1-dx2)/(2*h))
		}
		ms[i] = mtx.Axpy(-1*h, grad, ms[i])
		log.Printf("grad%d:\t%+v", i, grad)
	}
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
	bs := ms[3].Clone()
	return Branes{
		base:    ms,
		brn0:    brane.NewAffineBrane(ms[0].Clone(), bs.VGet(0)),
		brn1:    brane.NewSigmoidBrane(),
		brn2:    brane.NewAffineBrane(ms[1].Clone(), bs.VGet(1)),
		brn3:    brane.NewSigmoidBrane(),
		brn4:    brane.NewAffineBrane(ms[2].Clone(), bs.VGet(2)),
		brnLast: brane.NewSoftmaxCrossEntropyLossBrane(),
	}
}

func (bs *Branes) Predict(x mtx.Mtx) mtx.Mtx {
	//log.Printf(">x:\n%+.28v", x)
	o1 := bs.brn0.Forward(x)
	//log.Printf(">o1:\n%+.32v", o1)
	o2 := bs.brn1.Forward(o1)
	//log.Printf(">o2:\n%+.32v", o2)
	o3 := bs.brn2.Forward(o2)
	//log.Printf(">o3:\n%+.25v", o3)
	o4 := bs.brn3.Forward(o3)
	//log.Printf(">o4:\n%+.25v", o4)
	o := bs.brn4.Forward(o4)
	//log.Printf(">o:\n%+v", o)
	return o
}

func (bs *Branes) Reasoning() mtx.Mtx {
	dout := bs.brnLast.Backward()
	//log.Printf(">dout:\n%+.10v", dout)
	dx4 := bs.brn4.Backward(dout)
	//log.Printf(">dx4:\n%+v", dx4)
	dx3 := bs.brn3.Backward(dx4)
	//log.Printf(">dx3:\n%+v", dx3)
	dx2 := bs.brn2.Backward(dx3)
	//log.Printf(">dx2:\n%+v", dx2)
	dx1 := bs.brn1.Backward(dx2)
	//log.Printf(">dx1:\n%+v", dx1)
	dx0 := bs.brn0.Backward(dx1)
	//log.Printf(">dx0:\n%+.28v", dx0)
	return dx0
}

func (bs *Branes) Loss(x mtx.Mtx, t mtx.Mtx) float64 {
	o := bs.Predict(x)
	return bs.brnLast.Forward(o, t)
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

func GradientDescent(input mtx.Mtx, t []int, ms []mtx.Mtx, stepNum int, rate float64) []mtx.Mtx {
	linput, lt := get(input, t, 1)
	//log.Println(lt)
	for i := 0; i < stepNum; i++ {
		bs := NewBranes(ms)
		r := bs.Loss(linput, lt)
		log.Println("loss", r)
		bs.Reasoning()
		grads := bs.GetGrads()
		for i, g := range grads {
			ms[i] = mtx.Axpy(-1*rate, g, ms[i])
		}
		saveM(ms)
		//o := bs.Predict(linput)
		//log.Println(brane.MaxIndex(o))
	}
	return ms
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
	t := mtx.NewMtx(mtx.Shape{nums, 10})
	for i, v := range label {
		t.Set(i, v, 1.0)
	}
	return input, t
}
