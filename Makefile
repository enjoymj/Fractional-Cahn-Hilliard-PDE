ch2d: ch2d.c fft.c vec_add.c cl-helper.c frhs.c chstep.c chcg.c energy.c reduction.c
	gcc -std=gnu99 -D_XOPEN_SOURCE=500 -o$@ $^ -lOpenCL -lrt -lm


