package mtx

import "github.com/visionom/vision-mg/gems/blas"

func Plus(a, b Mtx) Mtx {
	return Xpy(a, b)
}

func Xpy(a, b Mtx) Mtx {
	a.SetData(blas.Daxpy(1.0, a.GetData(), b.GetData()))
	return a
}

func Ax(a float64, x Mtx) Mtx {
	y := NewMtx(x.Shape)
	x.SetData(blas.Daxpy(a, x.GetData(), y.GetData()))
	return x
}

func Axpy(a float64, x Mtx, y Mtx) Mtx {
	x.SetData(blas.Daxpy(a, x.GetData(), y.GetData()))
	return x
}
