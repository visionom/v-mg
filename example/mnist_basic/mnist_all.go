package main

import (
	"fmt"
	"math"
	"time"

	"github.com/visionom/vision-mg/gems/brane"
	"github.com/visionom/vision-mg/gems/mtx"
)

func main() {
	TrainData := readImage("./MNIST_data/train-images-idx3-ubyte")
	TrainLabel := readLabel("./MNIST_data/train-labels-idx1-ubyte")
	lens := 100
	list := make([]int, lens)
	for i := range list {
		list[i] = i
	}

	input := TrainData.GetRows(list)
	label := TrainLabel[0:lens]
	fmt.Printf("%+v\n", label)

	w1 := brane.Normpdf(mtx.NewMtx(mtx.Shape{784, 16}), 1.0, 1.0)
	wo := brane.Normpdf(mtx.NewMtx(mtx.Shape{16, 10}), 1.0, 1.0)
	b := mtx.NewMtx(mtx.Shape{1, 2})
	b.VSet(0, -4.0)
	b.VSet(1, -1.0)
	if true {
		ms := readM()
		w1 = ms[0]
		wo = ms[1]
		b = ms[2]
	}
	l := loss(input, initT(label), w1, wo, b)
	fmt.Printf("%v\n", l)
	gradientDescent(input, initT(label), w1, wo, b)
}

func initT(label []int) mtx.Mtx {
	m := mtx.NewMtx(mtx.Shape{len(label), 10})
	for i, x := range label {
		m.Set(i, x, 1.0)
	}
	return m
}

func predict(input, w1, wo, b mtx.Mtx) mtx.Mtx {
	o := model(input, w1, wo, b)
	return getMax(o)
}

func loss(input, label, w1, wo, b mtx.Mtx) float64 {
	o := model(input.Clone(), w1.Clone(), wo.Clone(), b.Clone())
	so := softmax(o.Clone())
	loss := crossError(so.Clone(), label.Clone())
	return loss
}

func diff(input, label mtx.Mtx, ms []mtx.Mtx, n int, h float64) (rm mtx.Mtx) {
	rm = mtx.NewMtx(ms[n].Shape)
	l := float64(len(ms[n].GetData()))
	for i, x := range ms[n].GetData() {
		t := x
		ms[n].VSet(i, t+h)
		dx1 := loss(input, label, ms[0], ms[1], ms[2])
		ms[n].VSet(i, t-h)
		dx2 := loss(input, label, ms[0], ms[1], ms[2])
		ms[n].VSet(i, t)
		dx := (dx1 - dx2) / (2 * h)
		rm.VSet(i, dx)
		fmt.Printf("\r%f", float64(i)/l)
	}
	return
}

func gradientDescent(input, label, w1, wo, b mtx.Mtx) {
	rate := 0.1
	ms := make([]mtx.Mtx, 3)
	ms[0] = w1.Clone()
	ms[1] = wo.Clone()
	ms[2] = b.Clone()
	for i := 0; i < 1000; i++ {
		sTime := time.Now()
		for j, m := range ms {
			g := diff(input, label, ms, j, rate)
			//			fmt.Printf("%+v", g)
			ms[j] = mtx.Axpy(-1*rate, g, m)
			fmt.Println(loss(input, label, ms[0], ms[1], ms[2]))
		}
		eTime := time.Now()
		o := predict(input, ms[0], ms[1], ms[2])
		fmt.Println(o.GetData(), eTime.Sub(sTime))
		saveM(ms)
	}
}

func model(input, w1, wo, b mtx.Mtx) mtx.Mtx {
	//fmt.Printf("%+.28v", input)
	//fmt.Printf("%+.28v", w1)
	h1 := mtx.MulBeta(input, w1, b.VGet(0))
	//fmt.Printf("%+.16v", h1)
	h2 := relu(h1)
	//fmt.Printf("%+.16v", h2)
	ho := mtx.MulBeta(h2, wo, b.VGet(1))
	return ho
}

func relu(a mtx.Mtx) mtx.Mtx {
	ra := mtx.NewMtx(a.Shape)
	for i, x := range a.GetData() {
		if x > 0 {
			ra.VSet(i, x)
		}
	}
	return ra
}

func softmax(a mtx.Mtx) mtx.Mtx {
	ra := mtx.NewMtx(a.Shape)
	for i := 0; i < ra.Shape[0]; i++ {
		r := a.GetRow(i)
		sum := 0.0
		for j, x := range r.GetData() {
			ra.Set(i, j, math.Exp(x))
			sum += ra.Get(i, j)
		}
		for j := range r.GetData() {
			ra.Set(i, j, ra.Get(i, j)/sum)
		}
	}
	return ra
}

func crossError(a, b mtx.Mtx) float64 {
	ra := mtx.NewMtx(mtx.Shape{a.Shape[0], 1})
	for i := 0; i < ra.Shape[0]; i++ {
		r := a.GetRow(i)
		for j, x := range r.GetData() {
			//fmt.Println(-1 * b.Get(i, j) * math.Log(x+1e-7))
			//ra.VSet(i, ra.VGet(i)-1*b.Get(i, j)*math.Log(x+1e-7))
			ra.VSet(i, ra.VGet(i)+math.Pow(b.Get(i, j)-x, 2))
			//fmt.Printf("%+v\n", -1*b.Get(i, j)*math.Log(x))
		}
	}
	//fmt.Printf("%+v", ra)
	sum := 0.0
	//lens := float64(len(ra.GetData()))
	for _, x := range ra.GetData() {
		//sum += (x + 1e-7) / lens
		sum += x
	}
	return sum / 2
}

func getMax(o mtx.Mtx) mtx.Mtx {
	//fmt.Println(o.Shape)
	ro := mtx.NewMtx(mtx.Shape{o.Shape[0], 1})
	for i := 0; i < o.Shape[0]; i++ {
		r := o.GetRow(i)
		max := o.VGet(0)
		maxi := 0
		for j, x := range r.GetData() {
			if max > x {
				max = x
				maxi = j
			}
		}
		ro.VSet(i, float64(maxi))
	}
	return ro
}
