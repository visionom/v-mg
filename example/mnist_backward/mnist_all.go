package main

import (
	"bufio"
	"flag"
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
	var (
		training bool
		_predict bool
		newModel bool
	)
	flag.BoolVar(&training, "training", false, "dump config sample and exit")
	flag.BoolVar(&_predict, "predict", false, "debug mode. Need read local config file")
	flag.BoolVar(&newModel, "new_model", false, "Local config file path")
	flag.Parse()

	go func() {
		log.Println(http.ListenAndServe("localhost:6060", nil))
	}()

	ms := make([]mtx.Mtx, 4)
	if newModel {
		ms[0] = brane.Normpdf(mtx.NewMtx(mtx.Shape{784, 1024}), 0.0, 1)
		ms[1] = brane.Normpdf(mtx.NewMtx(mtx.Shape{1024, 625}), 0.0, 0.1)
		ms[2] = brane.Normpdf(mtx.NewMtx(mtx.Shape{625, 10}), 0.0, 0.1)
		ms[3] = mtx.NewMtxNE(mtx.Shape{3, 1}, []float64{0, 0, 0})
	} else {
		ms = readM()
	}

	bs := &brane.Branes{
		InitBranes: newBranes,
		GetGrad:    getGrad,
	}

	if training {
		TrainData := readImage("./MNIST_data/train-images-idx3-ubyte")
		TrainLabel := readLabel("./MNIST_data/train-labels-idx1-ubyte")

		rawInput := TrainData
		rawLabel := TrainLabel

		gradientDescent(rawInput, rawLabel, ms, bs)
	}

	if _predict {
		TrainData := readImage("./MNIST_data/t10k-images-idx3-ubyte")
		TrainLabel := readLabel("./MNIST_data/t10k-labels-idx1-ubyte")

		rawInput := TrainData
		rawLabel := TrainLabel

		success := 0
		sum := 0

		for i, label := range rawLabel {
			input := rawInput.GetRow(i)
			o := bs.Predict(input, ms)
			o = getMax(o)
			sum++
			if int(o.VGet(0)) == label {
				success++
			}
			fmt.Printf("\r%f \t%d", float64(success)/float64(sum), i)
		}
	}
}

func initT(label []int) mtx.Mtx {
	m := mtx.NewMtx(mtx.Shape{len(label), 10})
	for i, x := range label {
		m.Set(i, x, 1.0)
	}
	return m
}

func getGrad(bs []brane.Brane) []mtx.Mtx {
	gs := make([]mtx.Mtx, 4)
	gs[3] = mtx.NewMtx(mtx.Shape{3, 1})
	for i, j := range []int{0, 2, 4} {
		if b, ok := bs[j].(*brane.AffineBrane); ok {
			gs[i] = b.Dw.Clone()
			gs[3].VSet(i, b.Db)
		} else {
			panic(b)
		}
	}
	return gs
}

func newBranes(ms []mtx.Mtx, last bool, label mtx.Mtx) []brane.Brane {
	b0 := brane.NewAffineBrane(ms[0].Clone(), ms[3].VGet(0))
	b1 := brane.NewReluBrane()
	b2 := brane.NewAffineBrane(ms[1].Clone(), ms[3].VGet(1))
	b3 := brane.NewReluBrane()
	b4 := brane.NewAffineBrane(ms[2].Clone(), ms[3].VGet(2))
	if last {
		b5 := brane.NewSoftmaxCrossEntropyLossBrane(label.Clone())
		return []brane.Brane{&b0, &b1, &b2, &b3, &b4, &b5}
	}
	return []brane.Brane{&b0, &b1, &b2, &b3, &b4}
}

func gradientDescent(rawInput mtx.Mtx, rawLabel []int, ms []mtx.Mtx, bs *brane.Branes) []mtx.Mtx {
	rate := 0.01
	for i := 0; i < 10000; i++ {
		input, label := getRandParams(rawInput, rawLabel, 100)
		sTime := time.Now()
		branes := bs.InitBranes(ms, true, label)
		loss := bs.Forward(input.Clone(), branes)
		//writeLoss(loss.VGet(0))
		bs.Backward(loss, branes)
		gs := bs.GetGrad(branes)
		for i, g := range gs {
			ms[i] = mtx.Axpy(-1*rate, g, ms[i].Clone())
		}

		eTime := time.Now()
		if i%1 == 0 {
			to := getMax(label)
			o := getMax(bs.Predict(input.Clone(), ms))
			sum := 0
			for i, v := range to.GetData() {
				if o.VGet(i) != v {
					sum++
				}
			}
			fmt.Println(to.GetData())
			fmt.Println(o.GetData())
			fmt.Println(sum, loss.GetData(), eTime.Sub(sTime))
			//saveM(ms)
		}
	}
	return ms
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

func getData(rawInput mtx.Mtx, rawLabel []int, i int) (input, label mtx.Mtx) {
	list := make([]int, 1)
	list[0] = rawLabel[i]
	return rawInput.GetRow(i), initT(list)
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
