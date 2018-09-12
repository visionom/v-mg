package blas

import (
	"reflect"
	"testing"
)

var DM1 = []float64{
	1, 2, 3,
	4, 5, 6,
	4, 5, 6,
	4, 5, 6,
}

var DM1T = []float64{
	1, 4, 4, 4,
	2, 5, 5, 5,
	3, 6, 6, 6,
}

var DM2 = []float64{
	1, 2,
	3, 4,
	5, 6,
}

var DM2T = []float64{
	1, 3, 5,
	2, 4, 6,
}

var DM3 = []float64{
	1, 2,
	3, 4,
	5, 6,
	6, 7,
}

var DM4 = []float64{
	1, 2, 3,
	4, 5, 6,
	7, 8, 9,
}

var DM5 = []float64{
	1, 2,
	3, 4,
	3, 4,
	3, 4,
}

func TestDgemm(t *testing.T) {
	type args struct {
		transA rune
		transB rune
		m      int
		n      int
		k      int
		alpha  float64
		a      []float64
		lda    int
		b      []float64
		ldb    int
		beta   float64
		c      []float64
		ldc    int
	}
	tests := []struct {
		name   string
		args   args
		wantRc []float64
	}{
		{
			"1",
			args{
				'N', 'N',
				4, 2, 3, 1,
				DM1, 1,
				DM2, 1,
				1,
				DM5, 1,
			},
			[]float64{
				23, 30,
				52, 68,
				52, 68,
				52, 68,
			},
		}, {
			"2",
			args{
				'T', 'N',
				4, 2, 3, 1,
				DM1T, 1,
				DM2, 1,
				1,
				DM5, 1,
			},
			[]float64{
				23, 30,
				52, 68,
				52, 68,
				52, 68,
			},
		}, {
			"3",
			args{
				'N', 'T',
				4, 2, 3, 1,
				DM1, 1,
				DM2T, 1,
				1,
				DM5, 1,
			},
			[]float64{
				23, 30,
				52, 68,
				52, 68,
				52, 68,
			},
		}, {
			"4",
			args{
				'T', 'T',
				4, 2, 3, 1,
				DM1T, 1,
				DM2T, 1,
				1,
				DM5, 1,
			},
			[]float64{
				23, 30,
				52, 68,
				52, 68,
				52, 68,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if gotRc := Dgemm(tt.args.transA, tt.args.transB, tt.args.m, tt.args.n, tt.args.k, tt.args.alpha, tt.args.a, tt.args.lda, tt.args.b, tt.args.ldb, tt.args.beta, tt.args.c, tt.args.ldc); !reflect.DeepEqual(gotRc, tt.wantRc) {
				t.Errorf("Dgemm() = %v, want %v", gotRc, tt.wantRc)
			}
		})
	}
}
