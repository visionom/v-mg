package mtx

import (
	"github.com/visionom/vision-mg/gems/blas"
)

func Plus(a, b Mtx) Mtx {
	return Xpy(a, b)
}

func Xpy(a, b Mtx) Mtx {
	ra := a.Clone()
	ra.SetData(blas.Daxpy(1.0, a.GetData(), b.GetData()))
	return ra
}

func Ax(a float64, x Mtx) Mtx {
	y := NewMtx(x.Shape)
	rx := x.Clone()
	rx.SetData(blas.Daxpy(a, x.GetData(), y.GetData()))
	return rx
}

func Axpy(a float64, x Mtx, y Mtx) Mtx {
	rx := x.Clone()
	rx.SetData(blas.Daxpy(a, x.GetData(), y.GetData()))
	return rx
}

func Aopy(a float64, y Mtx) Mtx {
	ones := NewOnes(y.Shape)
	ry := y.Clone()
	ry.SetData(blas.Daxpy(a, ones.GetData(), y.GetData()))
	return ry
}
