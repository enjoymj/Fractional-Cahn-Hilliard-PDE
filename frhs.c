#include<math.h>
#include<stdio.h>
#include"cl-helper.h"


#include"ch.h"

void frhs(/*variable*/cl_mem temp,  /*result*/ cl_mem temp2, 
		 cl_mem temp3, cl_mem temp4,cl_mem temp9, 
		struct parameter* p_param, cl_kernel fft_init,cl_kernel fft1D,cl_kernel mat_trans,
		 cl_kernel vec_add, cl_command_queue queue)
{
	int N = p_param->N;


//to calculate u for temp  f for temp2
//f = param.epsilon*real(ifft2(fft2(u).*symbol.sharmonic)) - ...
//   1/param.epsilon*real(ifft2(fft2(u.^3-u).*symbol.nlap_s));



	// fft(u.^3-u) .*nlap_s
	fft_w_orig(temp,temp3,temp4,temp9,N,p_param->epsilon,1,
	p_param->s,fft_init,fft1D,mat_trans,queue);



	fft2D(temp3,temp2,temp4,temp9,N,fft_init,fft1D,mat_trans,queue,-1);
	//real(ifft2(fft2(u).*sharmonic));
	fft_shar(temp,temp3,temp4,temp9,N,p_param->epsilon,0,p_param->s,fft_init,fft1D,mat_trans,queue);


	fft2D(temp3,temp4,temp3,temp9,N,fft_init,fft1D,mat_trans,queue,-1);



	vec__add(temp2,temp4,temp2,-1/p_param->epsilon ,p_param->epsilon ,2*N*N,vec_add,queue);

		#if 0
	float test;
	CALL_CL_GUARDED(clFinish, (queue));
	CALL_CL_GUARDED(clEnqueueReadBuffer, (
        	queue, temp2, /*blocking*/ CL_TRUE, /*offset*/ 0,
       		sizeof(float), &test,
        	0, NULL, NULL));
	

		printf("in frhs and %f \n", test);		
	

	#endif

	
}
