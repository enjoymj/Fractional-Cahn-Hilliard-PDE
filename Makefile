test: test_cluster.c cl-helper.c 
	gcc -std=gnu99 -D_XOPEN_SOURCE=500 -o$@ $^ -lOpenCL -lrt -lm


