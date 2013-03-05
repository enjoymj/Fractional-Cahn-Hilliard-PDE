#include "cl-helper.h"
#include "timing.h"
#include<CL/cl.h>
#include "ppm.h"

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include "ch.h"
#ifndef PI
#define PI 3.14156265358979323846
#endif


#define Q 1
#define Nlaps2 2
#define Shar 3
#define Nlaps 4
#define X 5
#define Y 6

void fft_w_orig2(cl_mem a, cl_mem c, cl_mem b,cl_mem d, int N,float epsilon,float k,float s, cl_kernel init_big,
		cl_kernel clean,cl_kernel mat_trans,cl_kernel mat_trans_3D, cl_command_queue queue)
{

	fft2D_transpose(a, c, b, d,N, init_big,
		clean,mat_trans, mat_trans_3D, queue,1,1);


	mat__trans(b,c,N,mat_trans,queue,Nlaps,epsilon,k,s);
}

void main(int argc, char** argv)
{

	 	

	float s = atof(argv[1]);
	float epsilon = atof(argv[2]);
	int k = atoi(argv[3]);
    int  N  = pow(2,k);
	float h=2*M_PI/N;



	//N =1024; k= 10;
	float * a = (float *) malloc(sizeof(float)*N* N * 2);
	float * b = (float *) malloc(sizeof(float) *N*N * 2);
	float * c = (float *) malloc(sizeof(float) * N*N* 2);
	float p = 2*M_PI ;	


	for(int i =0;i<N;i++)
		for(int j =0;j<N;j++)
		{	
		
			a[2*(i*N+j)] =2*exp(sin((i+1)*h)+sin((j+1)*h)-2)+2.2*exp(-sin((i+1)*h)-sin((j+1)*h)-2)-1;
			//b[2*(i*N+j)] = pow(a[2*(i*N+j)],3)-a[2*(i*N+j)];
			//printf("%f\n",uu[2*(i*N+j)]);
			a[2*(i*N+j)+1] =0 ;
			//b[2*(i*N+j)+1] =0 ;
		}



#if 0
	for (int i =0; i< N*N; i++)
	{
		a[2*i] = sin(M_PI *3/N *i);
		//a[2*i] = 1;
		a[2*i+1] = 0;
		b[2*i] = 1;
		b[2*i+1] = 0;
	}
#endif

#if 0 
	srand(1);
	for(int i =0;i<N*N;i++)
	{	
		a[2*i]=sin(i%N *2 *M_PI);
		//printf("%f\n",uu[2*i]);
		a[2*i+1] =0 ;
	}
#endif
	print_platforms_devices();

	cl_context ctx;
	cl_command_queue queue;
	create_context_on("Advanced Micro Devices","Turk",0,&ctx,&queue,0);

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

	cl_mem buf_f = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 
	sizeof(float) *N *N* 2 , 0, &status);
	CHECK_CL_ERROR(status, "clCreateBuffer");

	cl_mem buf_g = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 
	sizeof(float) *N *N* 2 , 0, &status);
	CHECK_CL_ERROR(status, "clCreateBuffer");

	cl_mem buf_h = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 
	sizeof(float) *N *N* 2 , 0, &status);
	CHECK_CL_ERROR(status, "clCreateBuffer");

	CALL_CL_GUARDED(clEnqueueWriteBuffer, (
	queue, buf_a, /*blocking*/ CL_TRUE, /*offset*/ 0,
	sizeof(float) *N*N*2, a,
	0, NULL, NULL));

	CALL_CL_GUARDED(clEnqueueWriteBuffer, (
	queue, buf_f, /*blocking*/ CL_TRUE, /*offset*/ 0,
	sizeof(float) *N *N* 2, b,
	0, NULL, NULL));

	CALL_CL_GUARDED(clEnqueueWriteBuffer, (
	queue, buf_c, /*blocking*/ CL_TRUE, /*offset*/ 0,
	sizeof(float)  *N* N*2, c,
	0, NULL, NULL));

	char *knl_text = read_file("vec_add.cl");
	cl_kernel vec_add = kernel_from_string(ctx, knl_text, "sum", NULL);
	free(knl_text);

	knl_text = read_file("mat_etr_mul.cl");
	cl_kernel mat_etr_mul = kernel_from_string(ctx, knl_text, "mult", NULL);
	free(knl_text);


	knl_text = read_file("radix-4-float.cl");
	cl_kernel fft1D = kernel_from_string(ctx, knl_text, "fft1D", NULL);
	free(knl_text);

	knl_text = read_file("radix-4-init.cl");
	cl_kernel fft_init = kernel_from_string(ctx, knl_text, "fft1D_init", NULL);
	free(knl_text);

	knl_text = read_file("radix-4-interm.cl");
	cl_kernel fft_interm = kernel_from_string(ctx, knl_text, "fft1D", NULL);
	free(knl_text);

	knl_text = read_file("transpose-soln-gpu.cl");
	cl_kernel mat_trans = kernel_from_string(ctx, knl_text, "transpose", NULL);
	free(knl_text);

	knl_text = read_file("radix-4-modi.cl");
	cl_kernel fft_init_w = kernel_from_string(ctx, knl_text, "fft1D_init", NULL);
	free(knl_text);

	knl_text = read_file("vec_zero.cl");
	cl_kernel vec_zero = kernel_from_string(ctx, knl_text, "zero", NULL);
	free(knl_text);

	knl_text = read_file("reduction.cl");
	cl_kernel reduct_mul = kernel_from_string(ctx, knl_text, "reduction_mult", NULL);
	free(knl_text);

	knl_text = read_file("reduction1D.cl");
	cl_kernel reduct = kernel_from_string(ctx, knl_text, "reduction", NULL);
	free(knl_text);

	knl_text = read_file("reduction-init.cl");
	cl_kernel reduct_init = kernel_from_string(ctx, knl_text, "reduction_init", NULL);
	free(knl_text);


	knl_text = read_file("reduct-energy.cl");
	cl_kernel reduct_eng = kernel_from_string(ctx, knl_text, "reduction_eng", NULL);
	free(knl_text);

	knl_text = read_file("resid.cl");
	cl_kernel resid = kernel_from_string(ctx, knl_text, "resid", NULL);
	free(knl_text);

	knl_text = read_file("resid-init.cl");
	cl_kernel resid_init = kernel_from_string(ctx, knl_text, "resid_init", NULL);
	free(knl_text);


	knl_text = read_file("radix-4-big.cl");
	cl_kernel fft_big = kernel_from_string(ctx, knl_text, "fft1D_big", NULL);
	free(knl_text);
	knl_text = read_file("radix-4-big-clean.cl");
	cl_kernel fft_clean = kernel_from_string(ctx, knl_text, "fft1D_clean", NULL);
	free(knl_text);

	knl_text = read_file("radix-4-2D.cl");
	cl_kernel fft_2D = kernel_from_string(ctx, knl_text, "fft2D_big", NULL);
	free(knl_text);

	knl_text = read_file("radix-4-2D-clean.cl");
	cl_kernel fft_2D_clean = kernel_from_string(ctx, knl_text, "fft2D_clean", NULL);
	free(knl_text);


	knl_text = read_file("mat-trans-3D.cl");
	cl_kernel mat_trans_3D = kernel_from_string(ctx, knl_text, "transpose_3D", NULL);
	free(knl_text);
	int Ns =1 ;
	int direction = 1;
	timestamp_type time1, time2;
	
	struct parameter param;

	param.N = N;
	param.epsilon = 0.1;
	param.s =1;
	
	float kk =1e-4;



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
	float elapsed ;
	int T;
	CALL_CL_GUARDED(clFinish, (queue));
	get_timestamp(&time1);



	CALL_CL_GUARDED(clFinish, (queue));
	get_timestamp(&time1);
	//fft2D_big_new(buf_a,buf_b,buf_c,buf_d,N,fft_2D,fft_2D_clean,
			//mat_trans,mat_trans_3D,queue,direction);
			frhs(buf_a,buf_b,buf_c,buf_d,buf_e,buf_f,&param,fft_2D,fft_2D_clean,
				mat_trans,mat_trans_3D,vec_add,queue);
//fft2D_transpose(buf_f,buf_c,buf_d,buf_e,N,fft_2D,fft_2D_clean,mat_trans,mat_trans_3D,queue,direction,0);

	//mat__trans(buf_d,buf_g,N,mat_trans,queue,4,epsilon,k,s);
	//fft_w_orig(buf_a,buf_b,buf_c,buf_d,N,fft_2D,fft_2D_clean,mat_trans,mat_trans_3D,queue,direction);
	CALL_CL_GUARDED(clFinish, (queue));
	get_timestamp(&time2);
	elapsed = timestamp_diff_in_seconds(time1,time2);
	printf("Using 2D kernel 2D FFT of size %d * %d matrix  on gpu takes %f s\n", N,N,elapsed);
	printf("achieve %f GFLOPS \n",5*2*N*N*k/elapsed*1e-9);
	printf("---------------------------------------------\n");



	get_timestamp(&time1);

	vec__add(buf_a, buf_b, buf_c, 1, kk , 
		2*N*N, vec_add, queue);

//fft2D(buf_c,buf_g,buf_f,buf_d,N,fft_2D,fft_2D_clean,mat_trans,mat_trans_3D,queue,1);
	//frhs(buf_c,buf_d,buf_e,buf_f,buf_g,buf_h,&param,fft_2D,fft_2D_clean,
				//mat_trans,mat_trans_3D,vec_add,queue);

	fft_w_orig(buf_a,buf_e,buf_f,buf_g,N,param.epsilon,1,
	param.s,fft_2D,fft_2D_clean,mat_trans,mat_trans_3D,queue);


	fft2D(buf_e,buf_g,buf_f,buf_d,N,fft_2D,fft_2D_clean,mat_trans,mat_trans_3D,queue,-1);





	//real(ifft2(fft2(u).*sharmonic));
	//fft_shar(buf_c,buf_e,buf_d,buf_f,N,param.epsilon,0,param.s,
//fft_2D,fft_2D_clean,mat_trans,mat_trans_3D,queue);


 #if 0
	fft2D_transpose(buf_a, buf_e,buf_d,buf_f,N, fft_2D,
		fft_2D_clean,mat_trans, mat_trans_3D, queue,1,0);

	mat__trans(buf_d,buf_e,N,mat_trans,queue,Shar,epsilon,k,s);


	fft2D(buf_e,buf_f,buf_d,buf_h,N,fft_2D,fft_2D_clean,mat_trans,mat_trans_3D,queue,-1);





//vec__add(buf_g,buf_f,buf_d,1,-1,2*N*N,vec_add,queue);
#endif
	direction = -1;
	//fft_1D(buf_b,buf_c,buf_d,N*N,fft_init, fft1D,queue,direction,0);
	//fft_1D_big(buf_b,buf_c,buf_d,N*N,fft_big,fft_clean,mat_trans,queue,direction,0);
	//fft2D(buf_b,buf_c,buf_d,buf_e,N,fft_init,fft1D,mat_trans,queue, direction);
	//fft2D(buf_g,buf_c,buf_d,buf_e,N,fft_2D,fft_2D_clean,mat_trans,mat_trans_3D,queue,direction);
	//fft2D(buf_e,buf_f,buf_d,buf_e,N,fft_2D,fft_2D_clean,mat_trans,mat_trans_3D,queue,direction);
	//fft2D_new(buf_b,buf_c,buf_e,buf_d,N,fft_init,fft_interm,fft1D,mat_trans,queue, -1);
	//fft2D_big(buf_b,buf_c,buf_d,buf_e,N,fft_big,fft_clean,mat_trans,queue,direction);
	CALL_CL_GUARDED(clFinish, (queue));
	get_timestamp(&time2);
	elapsed = timestamp_diff_in_seconds(time1,time2);
	//printf("1D inverse %f s\n", elapsed);
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
        	queue, buf_g, /*blocking*/ CL_TRUE, /*offset*/ 0,
       		2*N*N* sizeof(float), c,
        	0, NULL, NULL));
	

	/*for(int i =0; i<  N; i++)
	{
		printf("a%f+ i*",a[2*i]);		
		printf("%f\n",a[2*i+1]);
	}*/
	T = 10<N? 10:N ;
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
	CALL_CL_GUARDED(clReleaseKernel, (fft_init));
	CALL_CL_GUARDED(clReleaseKernel, (vec_add));
	CALL_CL_GUARDED(clReleaseKernel, (reduct_mul));
	CALL_CL_GUARDED(clReleaseKernel, (reduct));
	CALL_CL_GUARDED(clReleaseKernel, (mat_trans));
	CALL_CL_GUARDED(clReleaseCommandQueue, (queue));
	CALL_CL_GUARDED(clReleaseContext, (ctx));

}


