package main

import (
	"compress/gzip"
	"encoding/binary"
	"log"
	"os"

	"github.com/visionom/vision-mg/gems/mtx"
)

func readImage(path string) mtx.Mtx {
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

	images := mtx.NewMtx(mtx.Shape{int(imgNum), int(cols * rows)})
	k := 0
	for j := 0; j < int(imgNum); j++ {
		bs = make([]byte, int(cols*rows))
		n, err = gr.Read(bs)
		for _, b := range bs {
			images.VSet(k, float64(b)/255.0)
			k++
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
	for i := 0; i < int(nums); i++ {
		bs = make([]byte, 1)
		n, err = gr.Read(bs)
		labels[i] = int(bs[0])
	}
	return labels
}
