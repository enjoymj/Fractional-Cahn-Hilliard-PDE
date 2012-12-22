CL_INC = -I$(OPENCL_INC)
CL_LIB = -L$(OPENCL_LIB)

test: test_cluster.c cl-helper.c 
	gcc $(CL_INC) $(CL_LIB) -std=gnu99 -D_XOPEN_SOURCE=500 -o$@ $^ -lOpenCL -lrt -lm


