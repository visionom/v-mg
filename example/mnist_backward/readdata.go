package main

import (
	"encoding/binary"
	"fmt"
	"log"
	"math"
	"os"

	"github.com/visionom/vision-mg/gems/mtx"
)

func readImage(path string) mtx.Mtx {
	f, err := os.Open(path)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	bs := make([]byte, 16)
	n, err := f.Read(bs)
	if n < 4 {
		log.Panicf("read images data fails, data is not enough")
	}
	magic := binary.BigEndian.Uint32(bs[0:4])
	if magic != 2051 {
		log.Panicf("read images data fails, magic number is wrong, magic: %d", magic)
	}
	imgNum := int(binary.BigEndian.Uint32(bs[4:8]))
	rows := int(binary.BigEndian.Uint32(bs[8:12]))
	cols := int(binary.BigEndian.Uint32(bs[12:16]))

	log.Printf("read images data %s\n"+
		" >>> magic number: %d \n"+
		" >>> image number: %d \n"+
		" >>> rows number: %d\n"+
		" >>> cols number: %d",
		path, magic, imgNum, rows, cols)

	images := mtx.NewMtx(mtx.Shape{imgNum, cols * rows})
	for j := 0; j < imgNum; j++ {
		bs = make([]byte, cols*rows)
		n, err = f.Read(bs)
		for i, b := range bs {
			v := float64(b) / (255.0 * 16)
			//ni := (i/(cols*4))*(cols/4) + i%cols/4
			//images.Set(j, ni, v+images.Get(j, ni))
			images.Set(j, i, v)
		}
		//	log.Printf("%+.7v", images.GetRow(j))
	}
	return images
}

func readLabel(path string) []int {
	f, err := os.Open(path)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	bs := make([]byte, 8)
	n, err := f.Read(bs)
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
	for i := 0; i < int(nums); i++ {
		bs = make([]byte, 1)
		n, err = f.Read(bs)
		labels[i] = int(bs[0])
	}
	return labels
}

func readM() []mtx.Mtx {
	f, _ := os.Open("./m.dat")
	defer f.Close()
	bs := make([]byte, 4)
	f.Read(bs)
	lens := binary.BigEndian.Uint32(bs)
	log.Println(lens)

	ms := make([]mtx.Mtx, lens)

	for i := 0; i < int(lens); i++ {
		bs = make([]byte, 8)
		f.Read(bs)
		s := mtx.Shape{
			int(binary.BigEndian.Uint32(bs[0:4])),
			int(binary.BigEndian.Uint32(bs[4:8])),
		}
		log.Println(s)
		ms[i] = mtx.NewMtx(s)

		for j := range ms[i].GetData() {
			bs = make([]byte, 8)
			f.Read(bs)
			ms[i].VSet(j, float64frombytes(bs))
		}
	}
	return ms
}

func writeLoss(loss float64) {
	f, _ := os.OpenFile("./loss", os.O_APPEND|os.O_WRONLY, 0600)
	defer f.Close()
	f.WriteString(fmt.Sprintf("%f\n", loss))
	f.Sync()
}

func saveM(ms []mtx.Mtx) {
	var data []byte
	data = append(data, int32Bits(len(ms))...)
	for _, m := range ms {
		data = append(data, int32Bits(m.Shape[0])...)
		data = append(data, int32Bits(m.Shape[1])...)
		for _, v := range m.GetData() {
			data = append(data, float64bytes(v)...)
		}
	}

	f, _ := os.Create("./m.dat")
	defer f.Close()
	f.Write(data)
	f.Sync()
}

func float64frombytes(bytes []byte) float64 {
	bits := binary.LittleEndian.Uint64(bytes)
	float := math.Float64frombits(bits)
	return float
}

func float64bytes(float float64) []byte {
	bits := math.Float64bits(float)
	bytes := make([]byte, 8)
	binary.LittleEndian.PutUint64(bytes, bits)
	return bytes
}

func int32Bits(v int) []byte {
	bs := make([]byte, 4)
	binary.BigEndian.PutUint32(bs, uint32(v))
	return bs
}
