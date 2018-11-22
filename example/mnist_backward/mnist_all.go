package main

import (
	"bufio"
	"fmt"
	"log"
	"net/http"
	_ "net/http/pprof"
	"os"
	"os/exec"
	"time"

	"github.com/visionom/vision-mg/gems/brane"
	"github.com/visionom/vision-mg/gems/mtx"
)

func main() {
	go func() {
		log.Println(http.ListenAndServe("localhost:6060", nil))
	}()

	TrainData := readImage("./MNIST_data/train-images-idx3-ubyte")
	TrainLabel := readLabel("./MNIST_data/train-labels-idx1-ubyte")
	w1 := brane.Normpdf(mtx.NewMtx(mtx.Shape{784, 625}), 0.0, 1)
	wo := brane.Normpdf(mtx.NewMtx(mtx.Shape{625, 10}), 0.0, 0.1)
	b := mtx.NewMtx(mtx.Shape{1, 2})
	b.VSet(0, 0.0)
	b.VSet(1, 0.0)
	if false {
		ms := readM()
		w1 = ms[0]
		wo = ms[1]
		b = ms[2]
	}

	lens := 100
	//5 0 4 1 9 2 1 3 1 4
	//3 5 3 6 1 7 2 8 6 9
	//4 0 9 1 1 2 4 3 2 7
	//3 8 6 9 0 5 6 0 7 6
	//1 8 7 9 3 9 8 5 9 3
	//3 0 7 4 9 8 0 9 4 1
	//4 4 6 0 4 5 6 1 0 0
	//1 7 1 6 3 0 2 1 1 7
	//9 0 2 6 7 8 3 9 0 4
	//6 7 4 6 8 0 7 8 3 1
	list := make([]int, lens)
	for i := range list {
		list[i] = i
	}

	input := TrainData.GetRows(list)
	label := TrainLabel[0:lens]

	fmt.Println(label)
	o := predict(input, w1, wo, b)
	fmt.Println(o.GetData())
	gradientDescent(input, initT(label), w1, wo, b)
}

type Brane interface {
	Forward(x mtx.Mtx) mtx.Mtx
	Backward(dout mtx.Mtx) mtx.Mtx
}

func initT(label []int) mtx.Mtx {
	m := mtx.NewMtx(mtx.Shape{len(label), 10})
	for i, x := range label {
		m.Set(i, x, 1.0)
	}
	return m
}

func predict(input, w1, wo, b mtx.Mtx) mtx.Mtx {
	//log.Println("Predict")
	b0 := brane.NewAffineBrane(w1.Clone(), b.VGet(0))
	b1 := brane.NewReluBrane()
	b2 := brane.NewAffineBrane(wo.Clone(), b.VGet(1))
	bs := []Brane{&b0, &b1, &b2}
	o := forward(input.Clone(), bs)
	return getMax(o)
}

func gradientDescent(input, label, w1, wo, b mtx.Mtx) {
	log.Println("Descent")
	rate := 0.01
	ms := make([]mtx.Mtx, 3)
	ms[0] = w1.Clone()
	ms[1] = wo.Clone()
	ms[2] = b.Clone()

	for i := 0; i < 1000; i++ {
		sTime := time.Now()
		b0 := brane.NewAffineBrane(ms[0].Clone(), ms[2].VGet(0))
		b1 := brane.NewReluBrane()
		b2 := brane.NewAffineBrane(ms[1].Clone(), ms[2].VGet(1))
		b3 := brane.NewSoftmaxCrossEntropyLossBrane(label.Clone())
		bs := []Brane{&b0, &b1, &b2, &b3}

		loss := forward(input.Clone(), bs)
		backward(loss, bs)
		ms[0] = mtx.Axpy(-1*rate, b0.Dw, ms[0].Clone())
		ms[1] = mtx.Axpy(-1*rate, b2.Dw, ms[1].Clone())
		ms[2] = mtx.Axpy(-1*rate, mtx.NewMtxNE(mtx.Shape{1, 2},
			[]float64{b0.Db, b2.Db}), ms[2].Clone())

		eTime := time.Now()
		to := getMax(label)
		o := predict(input.Clone(), ms[0].Clone(), ms[1].Clone(), ms[2].Clone())
		sum := 0
		for i, v := range to.GetData() {
			if o.VGet(i) != v {
				sum++
			}
		}
		clear()
		fmt.Println(to.GetData())
		fmt.Println(o.GetData())
		fmt.Println(sum, loss.GetData(), eTime.Sub(sTime))
		//saveM(ms)
	}
	o := predict(input.Clone(), ms[0].Clone(), ms[1].Clone(), ms[2].Clone())
	fmt.Println(o.GetData())
}

func forward(input mtx.Mtx, bs []Brane) mtx.Mtx {
	x := input.Clone()
	//fmt.Println("input x", x.GetData())
	for _, b := range bs {
		x = b.Forward(x.Clone())
		//wait()
		//	fmt.Println(i, "output x", x.GetData())
	}
	return x
}

func backward(dout mtx.Mtx, bs []Brane) {
	for i := len(bs); i > 0; i-- {
		dout = bs[i-1].Backward(dout.Clone())
	}
}

func getMax(o mtx.Mtx) mtx.Mtx {
	ro := mtx.NewMtx(mtx.Shape{o.Shape[0], 1})
	for i := 0; i < o.Shape[0]; i++ {
		r := o.GetRow(i)
		max := r.VGet(0)
		maxi := 0
		for j, x := range r.GetData() {
			if max < x {
				max = x
				maxi = j
			}
		}
		ro.VSet(i, float64(maxi))
	}
	return ro
}

func wait() {
	buf := bufio.NewReader(os.Stdin)
	fmt.Print("> ")
	buf.ReadBytes('\n')
}

func clear() {
	cmd := exec.Command("clear") //Linux example, its tested
	cmd.Stdout = os.Stdout
	cmd.Run()
}
