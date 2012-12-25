#include"ch.h"


float energy(cl_mem a, cl_mem b, cl_mem c,cl_mem d, cl_mem e,cl_mem f,float k, 
		struct parameter* p_param, cl_kernel init_big,cl_kernel clean,
		cl_kernel mat_trans,cl_kernel mat_trans_3D, cl_kernel reduct_eng, 
		cl_kernel reduct,cl_command_queue queue)
{
	int N = p_param->N;	
	fft_d_x(a,b,c,d,N,p_param->epsilon,k,p_param->s,init_big,clean,mat_trans,mat_trans_3D,queue);
	
	fft2D(b,c,d,e,N,init_big,clean,mat_trans,mat_trans_3D,queue,-1);
	
	fft_d_y(a,b,d,e,N,p_param->epsilon,k,p_param->s,init_big,clean,mat_trans,mat_trans_3D,queue);
	fft2D(b,d,e,f,N,init_big,clean,mat_trans,mat_trans_3D,queue,-1);
	

	float output = p_param->h * p_param->h * reduct_energy(c,d,a,e,N,p_param->epsilon,reduct_eng,reduct,queue);
	return output;
	
}


float reduct_energy(cl_mem a,cl_mem b, cl_mem c,cl_mem d, int N, float epsilon, 
			cl_kernel reduct_eng,cl_kernel reduct, cl_command_queue queue)
{
	int n = N ;
	float output;
//CALL_CL_GUARDED(clFinish, (queue));
		//printf("aha, n = %d\n",n);
	
	if(n > 128)
	{	
		SET_6_KERNEL_ARGS(reduct_eng, a, b, c, d,n,epsilon);
		size_t ldim[] = { 128 };
		size_t gdim[] = { n };
		CALL_CL_GUARDED(clEnqueueNDRangeKernel,
			(queue, reduct_eng,
			/*dimensions*/ 1, NULL, gdim, ldim,
			0, NULL, NULL));
		n /= 128;
			
		CALL_CL_GUARDED(clFinish, (queue));
		//printf("aha, n = %d\n",n);
		while(n>=128)
		{
			SET_2_KERNEL_ARGS(reduct, d, n);
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
			SET_2_KERNEL_ARGS(reduct, d, n);
			size_t ldim[] = { n };
			size_t gdim[] = { n };
			CALL_CL_GUARDED(clEnqueueNDRangeKernel,
				(queue, reduct,
				/*dimensions*/ 1, NULL, gdim, ldim,
				0, NULL, NULL));
		}
					
		CALL_CL_GUARDED(clEnqueueReadBuffer, (
        		queue, d, /*blocking*/ CL_TRUE, /*offset*/ 0,
       			sizeof(float), &output,
        		0, NULL, NULL));

	}
	else 
	{
		SET_6_KERNEL_ARGS(reduct_eng, a, b, c, d,n,epsilon);
		size_t ldim[] = { n };
		size_t gdim[] = { n };
		CALL_CL_GUARDED(clEnqueueNDRangeKernel,
			(queue, reduct_eng,
			/*dimensions*/ 1, NULL, gdim, ldim,
			0, NULL, NULL));
		CALL_CL_GUARDED(clEnqueueReadBuffer, (
        		queue, d, /*blocking*/ CL_TRUE, /*offset*/ 0,
       			sizeof(float), &output,
        		0, NULL, NULL));
	}
	CALL_CL_GUARDED(clFinish, (queue));
	//printf("reduct energy = %f\n",output);
	return output;
}
