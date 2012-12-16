#include "ch.h"
#include"cl-helper.h"

float reduction_mult(cl_mem a,cl_mem b, cl_mem c, int N, cl_kernel reduct_mul,cl_kernel reduct, cl_command_queue queue)
{
	int n = N ;
	float output;
//CALL_CL_GUARDED(clFinish, (queue));
		//printf("aha, n = %d\n",n);
	
	if(n > 128)
	{	
		SET_4_KERNEL_ARGS(reduct_mul, a, b, c, n);
		size_t ldim[] = { 128 };
		size_t gdim[] = { n };
		CALL_CL_GUARDED(clEnqueueNDRangeKernel,
			(queue, reduct_mul,
			/*dimensions*/ 1, NULL, gdim, ldim,
			0, NULL, NULL));
		n /= 128;
			
		CALL_CL_GUARDED(clFinish, (queue));
		//printf("aha, n = %d\n",n);
		while(n>=128)
		{
			SET_2_KERNEL_ARGS(reduct, c, n);
			size_t ldim[] = { 128 };
			size_t gdim[] = { n };
			CALL_CL_GUARDED(clEnqueueNDRangeKernel,
				(queue, reduct,
				/*dimensions*/ 1, NULL, gdim, ldim,
				0, NULL, NULL));
			n /= 128;
		CALL_CL_GUARDED(clFinish, (queue));
		//printf("aha, n = %d\n",n);

		}

		if(n != 1)
		{
			SET_2_KERNEL_ARGS(reduct, c, n);
			size_t ldim[] = { n };
			size_t gdim[] = { n };
			CALL_CL_GUARDED(clEnqueueNDRangeKernel,
				(queue, reduct,
				/*dimensions*/ 1, NULL, gdim, ldim,
				0, NULL, NULL));
		}
					
		CALL_CL_GUARDED(clEnqueueReadBuffer, (
        		queue, c, /*blocking*/ CL_TRUE, /*offset*/ 0,
       			sizeof(float), &output,
        		0, NULL, NULL));

	}
	else 
	{
		SET_4_KERNEL_ARGS(reduct_mul, a, b, c, n);
		size_t ldim[] = { n };
		size_t gdim[] = { n };
		CALL_CL_GUARDED(clEnqueueNDRangeKernel,
			(queue, reduct_mul,
			/*dimensions*/ 1, NULL, gdim, ldim,
			0, NULL, NULL));
		CALL_CL_GUARDED(clEnqueueReadBuffer, (
        		queue, c, /*blocking*/ CL_TRUE, /*offset*/ 0,
       			sizeof(float), &output,
        		0, NULL, NULL));
	}
	CALL_CL_GUARDED(clFinish, (queue));
	return output;
}


float reduction(cl_mem a,cl_mem b, int N,cl_kernel reduct_init,cl_kernel reduct, cl_command_queue queue)
{
	int n = N ;
	float output;
	if(n > 128)
	{	
		SET_3_KERNEL_ARGS(reduct_init, a, b, n);
		size_t ldim[] = { 128 };
		size_t gdim[] = { n };
		CALL_CL_GUARDED(clEnqueueNDRangeKernel,
			(queue, reduct_init,
			/*dimensions*/ 1, NULL, gdim, ldim,
			0, NULL, NULL));
		n /= 128;
		//printf("AHHAHHHHHAHHA!!!!!!\n");
		while(n>=128)
		{
			SET_2_KERNEL_ARGS(reduct, b, n);
			size_t ldim[] = { 128 };
			size_t gdim[] = { n };
			CALL_CL_GUARDED(clEnqueueNDRangeKernel,
				(queue, reduct,
				/*dimensions*/ 1, NULL, gdim, ldim,
				0, NULL, NULL));
			n /= 128;
		}
		if(n != 1)
		{
			SET_2_KERNEL_ARGS(reduct, b, n);
			size_t ldim[] = { n };
			size_t gdim[] = { n };
			CALL_CL_GUARDED(clEnqueueNDRangeKernel,
				(queue, reduct,
				/*dimensions*/ 1, NULL, gdim, ldim,
				0, NULL, NULL));
		}
		CALL_CL_GUARDED(clEnqueueReadBuffer, (
        		queue, b, /*blocking*/ CL_TRUE, /*offset*/ 0,
       			sizeof(float), &output,
        		0, NULL, NULL));
	}
	else 
	{
		SET_3_KERNEL_ARGS(reduct_init, a, b, n);
		size_t ldim[] = { n };
		size_t gdim[] = { n };
		CALL_CL_GUARDED(clEnqueueNDRangeKernel,
			(queue, reduct_init,
			/*dimensions*/ 1, NULL, gdim, ldim,
			0, NULL, NULL));
		CALL_CL_GUARDED(clEnqueueReadBuffer, (
        		queue, b, /*blocking*/ CL_TRUE, /*offset*/ 0,
       			sizeof(float), &output,
        		0, NULL, NULL));
	}
	return output;
}


float residual(cl_mem a, cl_mem b, cl_kernel resid, cl_kernel resid_init, cl_command_queue queue,int N)
{
	int n = N ;
	float output;
	if(n > 128)
	{	
		SET_3_KERNEL_ARGS(resid_init, a, b, n);
		size_t ldim[] = { 128 };
		size_t gdim[] = { n };
		CALL_CL_GUARDED(clEnqueueNDRangeKernel,
			(queue, resid_init,
			/*dimensions*/ 1, NULL, gdim, ldim,
			0, NULL, NULL));
		n /= 128;
		CALL_CL_GUARDED(clFinish, (queue));
		//printf("AHHAHHHHHAHHA!!!!!!\n");
		while(n>=128)
		{
			SET_2_KERNEL_ARGS(resid, b, n);
			size_t ldim[] = { 128 };
			size_t gdim[] = { n };
			CALL_CL_GUARDED(clEnqueueNDRangeKernel,
				(queue, resid,
				/*dimensions*/ 1, NULL, gdim, ldim,
				0, NULL, NULL));
			n /= 128;
		}
		if(n != 1)
		{
			SET_2_KERNEL_ARGS(resid, b, n);
			size_t ldim[] = { n };
			size_t gdim[] = { n };
			CALL_CL_GUARDED(clEnqueueNDRangeKernel,
				(queue, resid,
				/*dimensions*/ 1, NULL, gdim, ldim,
				0, NULL, NULL));
		}
		CALL_CL_GUARDED(clEnqueueReadBuffer, (
        		queue, b, /*blocking*/ CL_TRUE, /*offset*/ 0,
       			sizeof(float), &output,
        		0, NULL, NULL));
	}
	else 
	{
		SET_3_KERNEL_ARGS(resid_init, a, b, n);
		size_t ldim[] = { n };
		size_t gdim[] = { n };
		CALL_CL_GUARDED(clEnqueueNDRangeKernel,
			(queue, resid_init,
			/*dimensions*/ 1, NULL, gdim, ldim,
			0, NULL, NULL));
		CALL_CL_GUARDED(clEnqueueReadBuffer, (
        		queue, b, /*blocking*/ CL_TRUE, /*offset*/ 0,
       			sizeof(float), &output,
        		0, NULL, NULL));
	}
	return output;
}
