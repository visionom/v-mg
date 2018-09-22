package data

type Mtrx interface {
	M() int
	N() int
	Add(d *Mtrx) *Mtrx
	Sub(d *Mtrx) *Mtrx
	Mul(d *Mtrx) *Mtrx
	Dot(d *Mtrx) *Mtrx
	Div(d *Mtrx) *Mtrx
	PowX(d *Mtrx) *Mtrx
	Sqr(d *Mtrx) *Mtrx
	Sqrt(d *Mtrx) *Mtrx
	Exp(d *Mtrx) *Mtrx
	Log(d *Mtrx) *Mtrx
	Abs(d *Mtrx) *Mtrx
	Sum(d *Mtrx) *Mtrx
	T(d *Mtrx) *Mtrx
	Tr(d *Mtrx) *Mtrx
	Zip(d *Mtrx) *Mtrx
	UnZip(d *Mtrx) *Mtrx
}

type MtrxComplex128 struct {
	m    int
	n    int
	data [][]complex128
}
