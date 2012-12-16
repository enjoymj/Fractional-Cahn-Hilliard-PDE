#include<math.h>
#include<stdio.h>
#include"cl-helper.h"


void vec__add(float *a, float *b, float *c, float a_mult, float b_mult, 
		int n, cl_kernel knl, cl_command_queue queue)
{
	//SET_6_KERNEL_ARGS(knl, a, b, c, a_mult, b_mult, n);


		  CALL_CL_GUARDED(clSetKernelArg, (knl, 0, sizeof(a), &a)); 
  CALL_CL_GUARDED(clSetKernelArg, (knl, 1, sizeof(b), &b)); 
  CALL_CL_GUARDED(clSetKernelArg, (knl, 2, sizeof(c), &c));
 
  CALL_CL_GUARDED(clSetKernelArg, (knl, 3, sizeof(a_mult), &a_mult)); 

  CALL_CL_GUARDED(clSetKernelArg, (knl, 4, sizeof(b_mult), &b_mult)); 
  CALL_CL_GUARDED(clSetKernelArg, (knl, 5, sizeof(n), &n)); 

	size_t ldim[] = { 128 };
	size_t gdim[] = { ((n + ldim[0] - 1)/ldim[0])*ldim[0] };
	CALL_CL_GUARDED(clEnqueueNDRangeKernel,
			(queue, knl,
			/*dimensions*/ 1, NULL, gdim, ldim,
			0, NULL, NULL));
}

/*
void mat_etr_mul(float *a, float *b, float *c, 
		long n, cl_kernel knl, cl_command_queue queue)
{
	SET_4_KERNEL_ARGS(*p_knl, a, b, c, n);
	size_t ldim[] = { 128 };
	size_t gdim[] = { ((n + ldim[0] - 1)/ldim[0])*ldim[0] };
	CALL_CL_GUARDED(clEnqueueNDRangeKernel,
			(*p_queue, *p_knl,
			1, NULL, gdim, ldim,
			0, NULL, NULL));
}*/

void vec__zero(cl_mem a, int n, cl_kernel vec_zero, cl_command_queue queue)
{
	SET_2_KERNEL_ARGS(vec_zero, a, n);
	size_t ldim[] = { 128 };
	size_t gdim[] = { n };
	CALL_CL_GUARDED(clEnqueueNDRangeKernel,
			(queue, vec_zero,
			/*dimensions*/ 1, NULL, gdim, ldim,
			0, NULL, NULL));
}
