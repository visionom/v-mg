package main

import (
	"bufio"
	"fmt"
	"log"
	"math/rand"
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
	ms := readM()
	if false {
		ms[0] = brane.Normpdf(mtx.NewMtx(mtx.Shape{784, 625}), 0.0, 1)
		ms[1] = brane.Normpdf(mtx.NewMtx(mtx.Shape{625, 10}), 0.0, 0.1)
		ms[2] = mtx.NewMtx(mtx.Shape{1, 2})
		ms[2].VSet(0, 0.0)
		ms[2].VSet(1, 0.0)
	}

	lens := 100

	rawInput := TrainData
	rawLabel := TrainLabel

	gradientDescent(rawInput, rawLabel, ms)

	input, label := getRandParams(rawInput, rawLabel, lens)
	fmt.Println(getMax(label))
	o := predict(input, ms)
	fmt.Println(o.GetData())
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

func predict(input mtx.Mtx, ms []mtx.Mtx) mtx.Mtx {
	bs := newBranes(input.Clone(), mtx.Mtx{}, ms, false)
	o := forward(input.Clone(), bs)
	return getMax(o)
}

func getGrad(bs []Brane) []mtx.Mtx {
	gs := make([]mtx.Mtx, 3)
	gs[2] = mtx.NewMtx(mtx.Shape{2, 1})
	for i, j := range []int{0, 2} {
		if b, ok := bs[j].(*brane.AffineBrane); ok {
			gs[i] = b.Dw.Clone()
			gs[2].VSet(i, b.Db)
		} else {
			panic(b)
		}
	}
	return gs
}

func newBranes(input, label mtx.Mtx, ms []mtx.Mtx, last bool) []Brane {
	b0 := brane.NewAffineBrane(ms[0].Clone(), ms[2].VGet(0))
	b1 := brane.NewReluBrane()
	b2 := brane.NewAffineBrane(ms[1].Clone(), ms[2].VGet(1))
	if last {
		b3 := brane.NewSoftmaxCrossEntropyLossBrane(label.Clone())
		return []Brane{&b0, &b1, &b2, &b3}
	}
	return []Brane{&b0, &b1, &b2}
}

func gradientDescent(rawInput mtx.Mtx, rawLabel []int, ms []mtx.Mtx) []mtx.Mtx {
	rate := 0.01
	for i := 0; i < 1000; i++ {
		input, label := getRandParams(rawInput, rawLabel, 100)
		sTime := time.Now()
		bs := newBranes(input.Clone(), label.Clone(), ms, true)
		loss := forward(input.Clone(), bs)
		backward(loss, bs)
		gs := getGrad(bs)
		for i, g := range gs {
			ms[i] = mtx.Axpy(-1*rate, g, ms[i].Clone())
		}

		eTime := time.Now()
		to := getMax(label)
		o := predict(input.Clone(), ms)
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
		saveM(ms)
	}
	return ms
}

func forward(input mtx.Mtx, bs []Brane) mtx.Mtx {
	x := input.Clone()
	for _, b := range bs {
		x = b.Forward(x.Clone())
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

func getRandParams(rawInput mtx.Mtx, rawLabel []int, lens int) (input, label mtx.Mtx) {
	max := len(rawLabel)
	list := make([]int, lens)
	numlabel := make([]int, lens)
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	for i := range list {
		n := r.Int() % max
		list[i] = n
		numlabel[i] = rawLabel[n]
	}
	return rawInput.GetRows(list), initT(numlabel)
}
