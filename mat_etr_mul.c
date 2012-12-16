#include<math.h>
#include<stdio.h>
#include"cl-helper.h"


/*void mat_etr_mul(cl_mem a, cl_mem b, cl_mem c, 
		long n, cl_kernel knl, cl_command_queue queue)
{
	SET_4_KERNEL_ARGS(knl, a, b, c, n);
	size_t ldim[] = { 128 };
	size_t gdim[] = { n };
	CALL_CL_GUARDED(clEnqueueNDRangeKernel,
			(queue, knl,
			 1, NULL, gdim, ldim,
			0, NULL, NULL));
}*/

void vec_zero(cl_mem a, int N,cl_kernel knl,cl_command_queue queue)
{
	SET_4_KERNEL_ARGS(knl, a, N);
	size_t ldim[] = { 128 };
	size_t gdim[] = { N };
	CALL_CL_GUARDED(clEnqueueNDRangeKernel,
			(queue, knl,
			/*dimensions*/ 1, NULL, gdim, ldim,
			0, NULL, NULL));
}
