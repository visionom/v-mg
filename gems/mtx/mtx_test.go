package mtx

import (
	"reflect"
	"testing"
)

var m2x2 = Mtx{
	Shape{2, 2},
	[]float64{
		1, 2,
		4, 5,
	},
}

var m3x3 = Mtx{
	Shape{3, 3},
	[]float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	},
}

var m3x3c0 = Mtx{
	Shape{3, 1},
	[]float64{
		1,
		4,
		7,
	},
}

var m3x3c2 = Mtx{
	Shape{3, 1},
	[]float64{
		3,
		6,
		9,
	},
}

var m3x3r0 = Mtx{
	Shape{1, 3},
	[]float64{
		1, 2, 3,
	},
}

var m3x3r2 = Mtx{
	Shape{1, 3},
	[]float64{
		7, 8, 9,
	},
}

var m2x3 = Mtx{
	Shape{2, 3},
	[]float64{
		1, 2, 3,
		4, 5, 6,
	},
}

func TestMtx_VGet(t *testing.T) {
	type args struct {
		x int
	}
	tests := []struct {
		name string
		m    Mtx
		args args
		want float64
	}{
		{
			"vget",
			m3x3.Clone(),
			args{0},
			1.0,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.m.VGet(tt.args.x); got != tt.want {
				t.Errorf("Mtx.VGet() = %+v, want %+v", got, tt.want)
			}
		})
	}
}

func TestMtx_VSet(t *testing.T) {
	type args struct {
		x int
		d float64
	}
	tests := []struct {
		name string
		m    Mtx
		args args
	}{
		{
			"vset",
			m3x3.Clone(),
			args{0, 2.0},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.m.VSet(tt.args.x, tt.args.d)
			got := tt.m.GetData()
			if got[tt.args.x] != tt.args.d {
				t.Errorf("Mtx.VSet() got %+v, want %+v", got[tt.args.x], tt.args.d)
			}
		})
	}
}

func TestMtx_Get(t *testing.T) {
	type args struct {
		x int
		y int
	}
	tests := []struct {
		name string
		m    Mtx
		args args
		want float64
	}{
		{
			"get",
			m2x3.Clone(),
			args{0, 0},
			1.0,
		}, {
			"get",
			m2x3.Clone(),
			args{0, 2},
			3.0,
		}, {
			"get",
			m2x3.Clone(),
			args{1, 0},
			4.0,
		}, {
			"get",
			m2x3.Clone(),
			args{1, 2},
			6.0,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.m.Get(tt.args.x, tt.args.y); got != tt.want {
				t.Errorf("Mtx.Get() = %+v, want %+v", got, tt.want)
			}
		})
	}
}

func TestMtx_Set(t *testing.T) {
	type args struct {
		x int
		y int
		d float64
	}
	tests := []struct {
		name string
		m    Mtx
		args args
	}{
		{
			"set",
			m2x3.Clone(),
			args{0, 0, -1.0},
		}, {
			"set",
			m2x3.Clone(),
			args{0, 2, -1.0},
		}, {
			"set",
			m2x3.Clone(),
			args{1, 0, -1.0},
		}, {
			"set",
			m2x3.Clone(),
			args{1, 2, -1.0},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.m.Set(tt.args.x, tt.args.y, tt.args.d)
			if got := tt.m.Get(tt.args.x, tt.args.y); got != -1.0 {
				t.Errorf("Mtx.Get() = %+v, want %+v", got, -1.0)
			}
		})
	}
}

func TestMtx_GetCol(t *testing.T) {
	type args struct {
		n int
	}
	tests := []struct {
		name string
		m    Mtx
		args args
		want Mtx
	}{
		{
			"get col",
			m3x3.Clone(),
			args{0},
			m3x3c0.Clone(),
		},
		{
			"get col",
			m3x3.Clone(),
			args{2},
			m3x3c2.Clone(),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.m.GetCol(tt.args.n); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Mtx.GetCol() = %+v, want %+v", got, tt.want)
			}
		})
	}
}

func TestMtx_GetRow(t *testing.T) {
	type fields struct {
		Shape Shape
		data  []float64
	}
	type args struct {
		n int
	}
	tests := []struct {
		name string
		m    Mtx
		args args
		want Mtx
	}{
		{
			"get row",
			m3x3.Clone(),
			args{0},
			m3x3r0.Clone(),
		},
		{
			"get row",
			m3x3.Clone(),
			args{2},
			m3x3r2.Clone(),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.m.GetRow(tt.args.n); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Mtx.GetRow() = %+v, want %+v", got, tt.want)
			}
		})
	}
}

func TestMtx_SetCol(t *testing.T) {
	type args struct {
		n int
		a Mtx
	}
	tests := []struct {
		name string
		m    Mtx
		args args
	}{
		{
			"set col",
			m3x3.Clone(),
			args{
				0,
				m3x3c0.Clone(),
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.m.SetCol(tt.args.n, tt.args.a)
		})
	}
}

func TestMtx_SetRow(t *testing.T) {
	type fields struct {
		Shape Shape
		data  []float64
	}
	type args struct {
		n int
		a Mtx
	}
	tests := []struct {
		name string
		m    Mtx
		args args
	}{
		{
			"set row",
			m3x3.Clone(),
			args{
				0,
				m3x3c0.Clone(),
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.m.SetRow(tt.args.n, tt.args.a)
		})
	}
}
