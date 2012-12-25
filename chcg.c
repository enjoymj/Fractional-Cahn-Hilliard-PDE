#include<math.h>
#include<stdio.h>
#include<stdbool.h>
#include"cl-helper.h"


#include"ch.h"


void chcg(float k,struct parameter * p_param,  cl_mem temp,cl_mem rhs, cl_mem temp2, bool *fail,
		cl_mem temp3,cl_mem temp4,cl_mem temp5,cl_mem temp6,cl_mem temp7,cl_mem temp8,cl_mem temp9,cl_kernel fft_2D,
		cl_kernel fft_2D_clean, cl_kernel fft_init_w,cl_kernel vec_add, cl_kernel vec_zero,
		cl_kernel mat_trans,cl_kernel mat_trans_3D,cl_kernel reduct,cl_kernel reduct_init,
		cl_kernel reduct_mul,cl_kernel resid, cl_kernel resid_init,cl_command_queue queue)
{
	* fail = false;
	int N = p_param->N;
	// fft2(rk)./q & rk =rhs 
	fft_d_q(rhs,temp2,temp9,temp3,N,p_param->epsilon,k,
		p_param->s,fft_2D,fft_2D_clean,mat_trans,mat_trans_3D,queue);
	//CALL_CL_GUARDED(clFinish, (queue));
	//printf("I am here!\n");
	#if 0
	float test;
	CALL_CL_GUARDED(clFinish, (queue));
	CALL_CL_GUARDED(clEnqueueReadBuffer, (
        	queue, rhs, /*blocking*/ CL_TRUE, /*offset*/ 0,
       		sizeof(float), &test,
        	0, NULL, NULL));
	

		printf("test success and %f \n",test);		
	

	#endif
	fft2D(temp2,temp3,temp9,temp4,N,fft_2D,fft_2D_clean,mat_trans,mat_trans_3D, queue,-1);
//------------------------	
//temp3 = zk temp =unew 
//------------------------

	// linvzk = real(ifft2(fft2(zk)./nlap_s2));  real part;
	fft_d_nlaps2(temp3,temp2,temp9,temp4,N,p_param->epsilon,k,p_param->s,fft_2D,fft_2D_clean,mat_trans,mat_trans_3D,queue);
	#if 0
	float test;
	CALL_CL_GUARDED(clFinish, (queue));
	CALL_CL_GUARDED(clEnqueueReadBuffer, (
        	queue, temp2, /*blocking*/ CL_TRUE, /*offset*/ 0,
       		sizeof(float), &test,
        	0, NULL, NULL));
	

		printf("test success and temp2 %f \n",test);		
	

	#endif
		
	fft2D(temp2,temp4,temp9,temp5,N,fft_2D,fft_2D_clean,mat_trans,mat_trans_3D,queue,-1);
	
//-----------------------------
//linvzk = temp4
//-----------------------------


//-----------------------------------	
//pk = temp5, temp8 = rk
//-------------------------------------




	clEnqueueCopyBuffer(queue,temp3,temp5,
		0,0,
		sizeof(float)*N*N*2,0,NULL,NULL);
	//vec_copy(rhs,temp8,2*N*N,p_knl,queue);
	clEnqueueCopyBuffer(queue,rhs,temp8,
		0,0,
		sizeof(float)*N*N*2,0,NULL,NULL);

//------------------
//temp2 = xk
//--------------------
	vec__zero(temp2,(2*N*N),vec_zero,queue);

	CALL_CL_GUARDED(clFinish, (queue));
	//printf("I am here!\n");
	#if 0
	float test;
	CALL_CL_GUARDED(clFinish, (queue));
	CALL_CL_GUARDED(clEnqueueReadBuffer, (
        	queue, temp4, /*blocking*/ CL_TRUE, /*offset*/ 0,
       		sizeof(float), &test,
        	0, NULL, NULL));
	

		printf("test success and %f \n",test);		
	

	#endif
	float ipnew = reduction_mult(temp4, rhs,temp9,(N*N),reduct_mul,reduct,queue);
	
	float ipold;

	
	float reside = 1;
	int iter = 0;
	float cgalpha;
	float beta;

	
		
	
	while(reside > p_param->cgtol && iter < p_param->maxCG)
	{
		ipold = ipnew;
		//printf("ipold = %f\n",ipnew);
		//fft((3*u.^2 -1) .* pk).*nlap_s

		fft_w(temp, temp5, temp7,temp4,N,p_param->epsilon,k,p_param->s,fft_2D,fft_init_w,fft_2D_clean,mat_trans,mat_trans_3D,queue);





	
		fft2D(temp4, temp6,temp9,temp3,N,fft_2D,fft_2D_clean,mat_trans,mat_trans_3D,queue,-1);

	#if 0
		float test;
		CALL_CL_GUARDED(clFinish, (queue));
		CALL_CL_GUARDED(clEnqueueReadBuffer, (
		queue, temp6, /*blocking*/ CL_TRUE, /*offset*/ 0,
		sizeof(float), &test,
		0, NULL, NULL));


		printf("test success and Apk =  %f \n",test*k/p_param->epsilon);		


		#endif	
		//---------------------------
		// temp4 = pk + Apk1
		//---------------------------
		vec__add(temp5, temp6, temp4,1,k/p_param->epsilon,2*N*N,vec_add, queue);

    		//Apk2 = -alpha*k*epsilon*real(ifft2(fft2(pk).*sharmonic));
		fft_shar(temp5, temp6,temp9,temp7,N, p_param->epsilon,k,p_param->s,fft_2D,fft_2D_clean,mat_trans,mat_trans_3D,queue);
		fft2D(temp6, temp7,temp9,temp3,N,fft_2D,fft_2D_clean,mat_trans,mat_trans_3D,queue,-1);
		//temp4 = Apk 
		vec__add(temp4, temp7, temp4, 1, -k*p_param->epsilon, 2*N*N,vec_add, queue);

		//linvpk = real(ifft2(fft2(pk)./nlap_s2)); temp6 = linvpk
		fft_d_nlaps2(temp5,temp7,temp9,temp3,N,p_param->epsilon,k,p_param->s,fft_2D,fft_2D_clean,mat_trans,mat_trans_3D,queue);
		fft2D(temp7,temp6,temp9,temp3,N,fft_2D,fft_2D_clean,mat_trans,mat_trans_3D,queue,-1);

		if( ipold > 1e-9) 
			cgalpha = ipold / reduction_mult(temp6, temp4,temp9,N*N,reduct_mul,reduct,queue);
		else cgalpha =0;

		vec__add(temp2, temp5,temp2,1, cgalpha,2*N*N,vec_add,queue);

//---------------------
// update temp8 =rk
//---------------------

		vec__add(temp8 , temp4, temp8, 1, -cgalpha,2*N*N,vec_add,queue);

		//temp3 = zk
		fft_d_q(temp8,temp6,temp9,temp3,N,p_param->epsilon,k,p_param->s,fft_2D,fft_2D_clean,mat_trans,mat_trans_3D,queue);
		fft2D(temp6,temp3,temp9,temp4,N,fft_2D,fft_2D_clean,mat_trans,mat_trans_3D,queue,-1);

		//linvzk = real(ifft2(fft2(zk)./nlap_s2)); temp6 = linvzk
		fft_d_nlaps2(temp3,temp7,temp9,temp4,N,p_param->epsilon,k,p_param->s,fft_2D,fft_2D_clean,mat_trans,mat_trans_3D,queue);
		fft2D(temp7,temp6,temp9,temp4,N,fft_2D,fft_2D_clean,mat_trans,mat_trans_3D,queue,-1);
		
		ipnew = reduction_mult(temp8, temp6,temp9,N*N,reduct_mul,reduct,queue);
		CALL_CL_GUARDED(clFinish, (queue));
		//printf("ipnew = %f\n",ipnew);
	#if 0
		float test;
		CALL_CL_GUARDED(clFinish, (queue));
		CALL_CL_GUARDED(clEnqueueReadBuffer, (
		queue, temp8, /*blocking*/ CL_TRUE, /*offset*/ 0,
		sizeof(float), &test,
		0, NULL, NULL));


		printf("test success and %f \n",test);		


		#endif	

		if (ipold >1e-9)
		beta = ipnew /ipold;
		else beta =0;
		vec__add(temp3,temp5,temp5,1,beta,2*N*N,vec_add,queue);
				
//printf("In cg step here!\n");
		reside = residual(temp8,temp9,resid,resid_init,queue,N*N);
		CALL_CL_GUARDED(clFinish, (queue));
		//printf("cg residual is %f\n",reside);
		iter ++;
  		p_param->cgloc ++;
	}

	if( reside > p_param->cgtol)
	{
		printf("too many CG steps\n");
		*fail = true;
	}



}
