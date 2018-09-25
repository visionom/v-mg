package mtx

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

func (m *Mtx) SetCol(a Mtx, n int) {
	for i := 0; i < m.Shape[0]; i++ {
		m.Set(i, n, a.VGet(i))
	}
}

func (m *Mtx) SetRow(a Mtx, n int) {
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

func (m *Mtx) GetCols(cols []int) Mtx {
	lcols := len(cols)
	s := Shape{m.Shape[0], lcols}
	ra := NewMtx(s)
	for j, k := range cols {
		ra.SetCol(m.GetCol(k), j)
	}
	return ra
}

func (m *Mtx) GetRows(rows []int) Mtx {
	lrows := len(rows)
	s := Shape{lrows, m.Shape[1]}
	ra := NewMtx(s)
	for j, k := range rows {
		ra.SetRow(m.GetRow(k), j)
	}
	return ra
}
