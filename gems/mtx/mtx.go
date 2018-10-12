package mtx

import (
	"fmt"
)

type Shape = [2]int

type Mtx struct {
	Shape Shape
	data  []float64
}

func NewMtx(s Shape) Mtx {
	sum := 1
	for _, v := range s {
		sum *= v
	}
	return Mtx{
		s,
		make([]float64, sum),
	}
}

func NewOnes(s Shape) Mtx {
	sum := 1
	for _, v := range s {
		sum *= v
	}
	ones := Mtx{
		s,
		make([]float64, sum),
	}
	ones.SetAll(1.0)
	return ones
}

func (m Mtx) Format(s fmt.State, verb rune) {
	switch verb {
	case 'v':
		if s.Flag('+') {
			if n, ok := s.Precision(); ok {
				fmt.Fprintf(s, "Shape:\t%v %d\nData: ", m.Shape, n)
				for i, v := range m.GetData() {
					if i%n == 0 {
						fmt.Fprint(s, "\t[")
					}
					fmt.Fprintf(s, " %.1f", v)
					if i%n == n-1 {
						fmt.Fprint(s, "]\n")
					}

				}
			} else {
				fmt.Fprintf(s, "Shape:\t%v\nData: ", m.Shape)
				for i := 0; i < m.Shape[0]; i++ {
					fmt.Fprint(s, "\t[")
					for j := 0; j < m.Shape[1]; j++ {
						fmt.Fprintf(s, " %.4e", m.Get(i, j))
					}
					fmt.Fprint(s, "]\n")
				}
			}
			return
		} else {
			v1 := m.Get(0, 0)
			v2 := m.Get(0, m.Shape[1]-1)
			v3 := m.Get(m.Shape[0]-1, 0)
			v4 := m.Get(m.Shape[0]-1, m.Shape[1]-1)
			fmt.Fprintf(s, "Shape:\t%v\nData: ", m.Shape)
			fmt.Fprintf(s, "\t%.4e . . . %.4e\n", v1, v2)
			fmt.Fprintf(s, "\t    .      .         .       \n")
			fmt.Fprintf(s, "\t    .        .       .       \n")
			fmt.Fprintf(s, "\t    .          .     .       \n")
			fmt.Fprintf(s, "\t%.4e . . . %.4e\n", v3, v4)
		}
	case 's':
		fallthrough
	case 'q':
		fallthrough
	default:
	}
}

func (m *Mtx) Clone() Mtx {
	rm := NewMtx(m.Shape)
	data := make([]float64, len(m.data))
	copy(data, m.data)
	rm.SetData(data)
	return rm
}

func (m *Mtx) VGet(x int) float64 {
	return m.data[x]
}

func (m *Mtx) VSet(x int, d float64) {
	m.data[x] = d
}

func (m *Mtx) Get(x, y int) float64 {
	return m.data[x*m.Shape[1]+y]
}

func (m *Mtx) Set(x, y int, d float64) {
	m.data[x*m.Shape[1]+y] = d
}

func (m *Mtx) GetCol(n int) Mtx {
	s := Shape{m.Shape[0], 1}
	ra := NewMtx(s)
	for i := 0; i < s[0]; i++ {
		ra.Set(i, 0, m.Get(i, n))
	}
	return ra
}

func (m *Mtx) GetRow(n int) Mtx {
	s := Shape{1, m.Shape[1]}
	ra := NewMtx(s)
	for i := 0; i < s[1]; i++ {
		ra.Set(0, i, m.Get(n, i))
	}
	return ra
}

func (m *Mtx) SetCol(n int, a Mtx) {
	for i := 0; i < m.Shape[0]; i++ {
		m.Set(i, n, a.VGet(i))
	}
}

func (m *Mtx) SetRow(n int, a Mtx) {
	for i := 0; i < m.Shape[1]; i++ {
		m.Set(n, i, a.VGet(i))
	}
}

func (m *Mtx) GetData() []float64 {
	return m.data
}

func (m *Mtx) SetData(data []float64) {
	m.data = data
}

func (m *Mtx) SetAll(a float64) {
	lens := len(m.data)
	for i := 0; i < lens; i++ {
		m.data[i] = a
	}
}

func (m *Mtx) GetCols(cols []int) Mtx {
	lcols := len(cols)
	s := Shape{m.Shape[0], lcols}
	ra := NewMtx(s)
	for j, k := range cols {
		ra.SetCol(j, m.GetCol(k))
	}
	return ra
}

func (m *Mtx) GetRows(rows []int) Mtx {
	lrows := len(rows)
	s := Shape{lrows, m.Shape[1]}
	ra := NewMtx(s)
	for j, k := range rows {
		ra.SetRow(j, m.GetRow(k))
	}
	return ra
}
