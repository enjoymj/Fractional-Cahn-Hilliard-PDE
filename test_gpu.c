#include "cl-helper.h"
#include "timing.h"
#include<CL/cl.h>
#include "ppm.h"

#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#ifndef M_PI
#define M_PI 3.14156265358979323846
#endif

#include "ch.h"


void main(int argc, char** argv)
{
	int k = atoi(argv[1]);	
	int  N  = pow(2,k);

	
	float * a = (float *) malloc(sizeof(float)*N* N * 2);
	float * b = (float *) malloc(sizeof(float) *N*N * 2);
	float * c = (float *) malloc(sizeof(float) * N*N* 2);
	float p = 2*M_PI ;	
	for (int i =0; i< N*N; i++)
	{
		a[2*i] = 1;
		a[2*i+1] = 0;
		b[2*i] = 1;
		b[2*i+1] = 0;
	}

	srand(1);
	for(int i =0;i<N*N;i++)
	{	
		a[2*i]=sin(i%N *2 *PI);
		//printf("%f\n",uu[2*i]);
		a[2*i+1] =0 ;
	}
	print_platforms_devices();

	cl_context ctx;
	cl_command_queue queue;
	create_context_on("Advanced Micro Devices","AMD",0,&ctx,&queue,0);

	cl_context ctx1;
	cl_command_queue queue1;
	create_context_on("Advanced Micro Devices","AMD",0,&ctx1,&queue1,0);

	cl_int status;
	cl_mem buf_a = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 
	sizeof(float) *N *N* 2 , 0, &status);
	CHECK_CL_ERROR(status, "clCreateBuffer");

	cl_mem buf_b = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 
	sizeof(float)  * N *N* 2 , 0, &status);
	CHECK_CL_ERROR(status, "clCreateBuffer");
	
	cl_mem buf_c = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 
	sizeof(float) * N *N* 2 , 0, &status);
	CHECK_CL_ERROR(status, "clCreateBuffer");

	cl_mem buf_d = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 
	sizeof(float)*N *N* 2 , 0, &status);
	CHECK_CL_ERROR(status, "clCreateBuffer");
	cl_mem buf_e = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 
	sizeof(float) *N *N* 2 , 0, &status);
	CHECK_CL_ERROR(status, "clCreateBuffer");

	CALL_CL_GUARDED(clEnqueueWriteBuffer, (
	queue, buf_a, /*blocking*/ CL_TRUE, /*offset*/ 0,
	sizeof(float) *N*N*2, a,
	0, NULL, NULL));

	CALL_CL_GUARDED(clEnqueueWriteBuffer, (
	queue, buf_b, /*blocking*/ CL_TRUE, /*offset*/ 0,
	sizeof(float) *N *N* 2, b,
	0, NULL, NULL));

	CALL_CL_GUARDED(clEnqueueWriteBuffer, (
	queue, buf_c, /*blocking*/ CL_TRUE, /*offset*/ 0,
	sizeof(float)  *N* N*2, c,
	0, NULL, NULL));

	char *knl_text = read_file("radix-4-float.cl");
	cl_kernel fft1D = kernel_from_string(ctx, knl_text, "fft1D", NULL);
	free(knl_text);

	knl_text = read_file("radix-4-init.cl");
	cl_kernel fft1D_init = kernel_from_string(ctx, knl_text, "fft1D_init", NULL);
	free(knl_text);

	knl_text = read_file("transpose-soln-gpu.cl");
	cl_kernel mat_trans = kernel_from_string(ctx, knl_text, "transpose", NULL);
	free(knl_text);

	knl_text = read_file("reduction.cl");
	cl_kernel reduct_mul = kernel_from_string(ctx, knl_text, "reduction_mult", NULL);
	free(knl_text);

	knl_text = read_file("reduction1D.cl");
	cl_kernel reduct = kernel_from_string(ctx, knl_text, "reduction", NULL);
	free(knl_text);

	knl_text = read_file("vec_add.cl");
	cl_kernel vec_add = kernel_from_string(ctx, knl_text, "sum", NULL);
	free(knl_text);

	knl_text = read_file("resid.cl");
	cl_kernel resid = kernel_from_string(ctx, knl_text, "resid", NULL);
	free(knl_text);

	knl_text = read_file("resid-init.cl");
	cl_kernel resid_init = kernel_from_string(ctx, knl_text, "resid_init", NULL);
	free(knl_text);

	int Ns =1 ;
	int direction = 1;
	timestamp_type time1, time2;
	
	struct parameter param;

	param.N = N;
	param.epsilon = 0.1;
	param.s =1;
	
	k =1e-4;



	param.h = 2*PI/N;
	param.N = N;
	
 	param.maxCG = 1000;
	param.maxN = 5;
	
	//Minimum and starting time step
	float mink = 1e-7;
	float startk = 1e-4;

	// Tolerances
	param.Ntol = 1e-4;
	param.cgtol = 1e-7;
	float ksafety = 0.8;
	float kfact = 1.3;
	float kfact2 = 1/1.3;
	float Nfact = 0.7;
	float CGfact = 0.7;


	CALL_CL_GUARDED(clFinish, (queue));
	get_timestamp(&time1);
	//fft_1D(buf_a,buf_b,buf_c,N,fft1D_init, fft1D,queue,direction,0);
	//fft2D(buf_a,buf_b,buf_c,buf_d,N,fft1D_init,fft1D,mat_trans,queue, 1);
	//fft_w_orig(buf_a,buf_b,buf_c,buf_d,N,0.1,0,1,fft1D_init,fft1D,mat_trans,queue);
#if 1
	frhs(buf_a,buf_b,buf_c,buf_d,buf_e,&param,fft1D_init,fft1D,mat_trans,
		 vec_add, queue);
#endif
	
	//float reside = residual(buf_a,buf_b,resid,resid_init,queue,N*N);
	/*fft_d_q(buf_a,buf_b,buf_c,buf_d, N,0.1,k ,1, 
		 fft1D_init,
		fft1D,mat_trans,queue);*/
	//for(int j= 0;j<N;j++)
	//{
		//fft_1D_w_orig(buf_a,buf_b,buf_c,N,fft1D_init,fft1D,queue,1,j);
	//}
	//fft_shar(buf_a,buf_b,buf_c,buf_d,N,0.1,0,1,fft1D_init,fft1D,mat_trans,queue);
	//mat__trans(buf_a,buf_b,N,mat_trans,queue,4,0.1,0,1);
	//double elapsed = reduction_mult(buf_a, buf_b,buf_c,N*N,reduct_mul,reduct,queue);
	CALL_CL_GUARDED(clFinish, (queue));
	//printf("come on %f \n", reside);
	CALL_CL_GUARDED(clFinish, (queue));
	get_timestamp(&time2);
	double elapsed = timestamp_diff_in_seconds(time1,time2);
	printf("on gpu %f s\n", elapsed);
	printf("achieve %f GFLOPS \n",8*N*N*k/elapsed*1e-9);
	CALL_CL_GUARDED(clFinish, (queue));
	get_timestamp(&time1);
	direction = -1;
	//fft_1D(buf_b,buf_c,buf_d,N,fft1D_init, fft1D,queue,direction,0);
	//fft2D(buf_b,buf_c,buf_d,buf_e,N,fft1D_init,fft1D,mat_trans,queue, direction);
	CALL_CL_GUARDED(clFinish, (queue));
	get_timestamp(&time2);
	elapsed = timestamp_diff_in_seconds(time1,time2);
	printf("1D inverse %f s\n", elapsed);
	#if 0
	float test;
	CALL_CL_GUARDED(clFinish, (queue));
	CALL_CL_GUARDED(clEnqueueReadBuffer, (
        	queue, buf_b, /*blocking*/ CL_TRUE, /*offset*/ 0,
       		sizeof(float), &test,
        	0, NULL, NULL));
	

		printf("test success and %f \n",test);		
	

	#endif
	#if 1
	CALL_CL_GUARDED(clFinish, (queue));
	CALL_CL_GUARDED(clEnqueueReadBuffer, (
        	queue, buf_b, /*blocking*/ CL_TRUE, /*offset*/ 0,
       		2*N*N* sizeof(float), c,
        	0, NULL, NULL));
	

	/*for(int i =0; i<  N; i++)
	{
		printf("a%f+ i*",a[2*i]);		
		printf("%f\n",a[2*i+1]);
	}*/
	int T = 10<N? 10:N ;
	for(int i =0; i<  T; i++)
	{
		printf("%f + i*",a[2*i]);		
		printf("%f\t",a[2*i+1]);
		printf("%f + i*",c[2*i]);		
		printf("%f\n",c[2*i+1]);
	}

	#endif 
/*	for( Ns = 1;Ns < N; Ns *= 2 )
	{
		for (int j = 0; j<N/2; j++)
		{
			fftiteration(j,N,Ns,a,b);
		}
		float * d;
		d = a ;
		a = b;
		b = d;
		//printf("ok\n");

	}

*/


	
	CALL_CL_GUARDED(clReleaseMemObject, (buf_a));
	CALL_CL_GUARDED(clReleaseMemObject, (buf_b));
	CALL_CL_GUARDED(clReleaseMemObject, (buf_c));
	CALL_CL_GUARDED(clReleaseMemObject, (buf_d));
	CALL_CL_GUARDED(clReleaseMemObject, (buf_e));
	CALL_CL_GUARDED(clReleaseKernel, (fft1D));
	CALL_CL_GUARDED(clReleaseKernel, (fft1D_init));
	CALL_CL_GUARDED(clReleaseKernel, (vec_add));
	CALL_CL_GUARDED(clReleaseKernel, (reduct_mul));
	CALL_CL_GUARDED(clReleaseKernel, (reduct));
	CALL_CL_GUARDED(clReleaseKernel, (mat_trans));
	CALL_CL_GUARDED(clReleaseCommandQueue, (queue));
	CALL_CL_GUARDED(clReleaseContext, (ctx));

}


