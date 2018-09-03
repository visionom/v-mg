package main

import (
	"compress/gzip"
	"encoding/binary"
	"log"
	"os"
)

type MGMatrix struct {
	M    int
	N    int
	Data []float32
}

type MGSession struct {
	Ws []MGMatrix
	Bs []float32
}

type MG interface {
	ReadData()
	InitSession()
	Model()
	Loss()
	Test()
	Run()
}

type MnistMG struct {
	MGSession
	TrainData  []MGMatrix
	TrainLabel []int
	TestData   []MGMatrix
	Testlabel  []int
}

func readImage(path string) []MGMatrix {
	f, err := os.Open(path)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	gr, err := gzip.NewReader(f)
	if err != nil {
		log.Fatal(err)
	}
	defer gr.Close()
	bs := make([]byte, 16)
	n, err := gr.Read(bs)
	if n < 4 {
		log.Panicf("read images data fails, data is not enough")
	}
	magic := binary.BigEndian.Uint32(bs[0:4])
	if magic != 2051 {
		log.Panicf("read images data fails, magic number is wrong, magic: %d", magic)
	}
	imgNum := binary.BigEndian.Uint32(bs[4:8])
	rows := binary.BigEndian.Uint32(bs[8:12])
	cols := binary.BigEndian.Uint32(bs[12:16])

	log.Printf("read images data %s\n"+
		" >>> magic number: %d \n"+
		" >>> image number: %d \n"+
		" >>> rows number: %d\n"+
		" >>> cols number: %d",
		path, magic, imgNum, rows, cols)

	images := make([]MGMatrix, imgNum)
	for i := 0; i < int(imgNum); i++ {
		bs = make([]byte, rows*cols)
		n, err = gr.Read(bs)
		images[i] = MGMatrix{M: int(rows), N: int(cols), Data: make([]float32, 28*28)}
		for j, b := range bs {
			images[i].Data[j] = float32(b)
		}
	}
	return images
}

func readLabel(path string) []int {
	f, err := os.Open(path)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	gr, err := gzip.NewReader(f)
	if err != nil {
		log.Fatal(err)
	}
	defer gr.Close()
	bs := make([]byte, 16)
	n, err := gr.Read(bs)
	if n < 4 {
		log.Panicf("read label data fails, data is not enough")
	}
	magic := binary.BigEndian.Uint32(bs[0:4])
	if magic != 2049 {
		log.Panicf("read label data fails, magic number is wrong, magic: %d", magic)
	}
	nums := binary.BigEndian.Uint32(bs[4:8])

	log.Printf("read images data %s \n"+
		" >>> magic number: %d \n"+
		" >>> items number: %d \n",
		path, magic, nums)

	labels := make([]int, nums)
	bs = make([]byte, nums)
	n, err = gr.Read(bs)
	for i, b := range bs {
		labels[i] = int(b)
	}
	return labels
}

func (m *MnistMG) ReadData() {
	m.TrainData = readImage("./MNIST_data/train-images-idx3-ubyte.gz")
	m.TrainLabel = readLabel("./MNIST_data/train-labels-idx1-ubyte.gz")

	m.TestData = readImage("./MNIST_data/t10k-images-idx3-ubyte.gz")
	m.Testlabel = readLabel("./MNIST_data/t10k-labels-idx1-ubyte.gz")
}

func main() {
	m := MnistMG{}
	m.ReadData()
}
