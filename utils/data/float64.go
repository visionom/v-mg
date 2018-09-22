package data

//MtrxF8 means matrix of type float64, impl of Mtrx
type MtrxF8 struct {
	m    int
	n    int
	data [][]float64
}

//NewMtrxF8 returns empty m by n matrix of type float64
func NewMtrxF8(m, n int) Mtrx {
	data := make([][]float64, m*n)
	return newMtrxF8(m, n, data)
}

func newMtrxF8(m, n int, data [][]float64) Mtrx {
	return &MtrxF8{m, n, data}
}

func (m *MtrxF8) M() int              { return m.m }
func (m *MtrxF8) N() int              { return m.n }
func (m *MtrxF8) Add(d *Mtrx) *Mtrx   { return nil }
func (m *MtrxF8) Sub(d *Mtrx) *Mtrx   { return nil }
func (m *MtrxF8) Mul(d *Mtrx) *Mtrx   { return nil }
func (m *MtrxF8) Dot(d *Mtrx) *Mtrx   { return nil }
func (m *MtrxF8) Div(d *Mtrx) *Mtrx   { return nil }
func (m *MtrxF8) PowX(d *Mtrx) *Mtrx  { return nil }
func (m *MtrxF8) Sqr(d *Mtrx) *Mtrx   { return nil }
func (m *MtrxF8) Sqrt(d *Mtrx) *Mtrx  { return nil }
func (m *MtrxF8) Exp(d *Mtrx) *Mtrx   { return nil }
func (m *MtrxF8) Log(d *Mtrx) *Mtrx   { return nil }
func (m *MtrxF8) Abs(d *Mtrx) *Mtrx   { return nil }
func (m *MtrxF8) Sum(d *Mtrx) *Mtrx   { return nil }
func (m *MtrxF8) T(d *Mtrx) *Mtrx     { return nil }
func (m *MtrxF8) Tr(d *Mtrx) *Mtrx    { return nil }
func (m *MtrxF8) Zip(d *Mtrx) *Mtrx   { return nil }
func (m *MtrxF8) UnZip(d *Mtrx) *Mtrx { return nil }
