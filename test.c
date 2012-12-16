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


void mat__trans(cl_mem a, cl_mem b, int N, cl_kernel mat_trans, cl_command_queue queue)
{
	SET_3_KERNEL_ARGS(mat_trans, a, b, N);
	//int N = 1024;


	size_t ldim[] = { 16, 16 };
	size_t gdim[] = { N, N };
	CALL_CL_GUARDED(clEnqueueNDRangeKernel,
	(queue, mat_trans,
	/*dimensions*/ 2, NULL, gdim, ldim,
	0, NULL, NULL));
}

void VecCopy(cl_mem c,cl_mem b,int N,int offset_line,cl_kernel vec_copy,cl_command_queue queue,int flag)
{
	//if(offset)
}

void fft_1D(cl_mem a,cl_mem b,cl_mem c, int N, cl_kernel knl,cl_command_queue queue,int direction,int offset_line)
{
	//handle complex-to-complex fft, accutal size = 2 * N

	//size_t ldim[] = { 128 };
	//size_t gdim[] = { (N /ldim[0])/2};
	
	for(int Ns=1; Ns<N; Ns<<=1)
	{
		if (Ns == 1)		
		{
			SET_6_KERNEL_ARGS(knl, a, b, N, Ns,direction,offset_line);
			size_t ldim[] = { 1 };
			size_t gdim[] = { N/2 };

			CALL_CL_GUARDED(clEnqueueNDRangeKernel,
					(queue, knl,
					 1, NULL, gdim, ldim,
					0, NULL, NULL));
			//CALL_CL_GUARDED(clFinish, (queue));
			//printf("ok\n");
			
			
		}
		else 
		{
			SET_6_KERNEL_ARGS(knl, b, c, N, Ns,direction,offset_line);
			size_t ldim[] = { 1 };
			size_t gdim[] = { N/2 };
			
			CALL_CL_GUARDED(clEnqueueNDRangeKernel,
					(queue, knl,
					 1, NULL, gdim, ldim,
					0, NULL, NULL));
			clEnqueueCopyBuffer(queue,c,b,0,0,sizeof(double)*N*N*2,0,NULL,NULL);
			//VecCopy(c,b,N,offset_line,vec_copy,queue);
			
			
			
		}  
	}
	
}

/* implementation of transpose-split method for 2D FFT*/


void fft2D(cl_mem a, cl_mem b, cl_mem c,cl_mem d, int N, 
		cl_kernel fft1D,cl_kernel mat_trans, cl_command_queue queue,int direction)
{
	

	for(int j= 0;j<N;j++)
	{
		fft_1D(a,b,c,N,fft1D,queue,direction,j);
	}
	CALL_CL_GUARDED(clFinish, (queue));
	printf("1D fine \n");

	mat__trans(b,c,N,mat_trans,queue);

	CALL_CL_GUARDED(clFinish, (queue));
	for(int j= 0;j<N;j++)
	{
		fft_1D(b,c,d,N,fft1D,queue,direction,j);
	}
	CALL_CL_GUARDED(clFinish, (queue));

	mat__trans(c,b,N,mat_trans,queue);
	
	
}


void FFT2(double * v)
{
	double v0 = v[0];
	double v00 = v[1];
	v[0] = v0 + v[2];
	
	v[1] = v00 + v[3];
	v[2] = v0 - v[2];
	v[3] = v00 - v[3];
	
}
//Stockham radix-2 fft
void fftiteration(
	int j,
    int N,
    int Ns,
    double *a,
    double *b )
{
	
	int gid = j;
	
	//radix-2 fft	
	double v[4];
	int idxS = gid;
	double xx;
	double yy;
	double angle = -2 *M_PI*(gid % Ns)/(Ns * 2) ;
	for(int  r = 0; r < 2; r++)
	{
		xx = v[2*r] = a[2 * (idxS + r * N / 2)];
		yy = v[2*r+1] = a[2 * (idxS + r * N / 2)+1];

		v[2 * r] = xx* cos(r*angle)- yy* sin(r *angle); 
		v[2*r +1 ]= xx * sin(r * angle) + yy * cos(r * angle);
		
		
	}
	FFT2(v);
	int idxD = (gid / Ns)*Ns*2 + gid % Ns;

	
	for (int r =0; r< 2;r++)
	{
		b[2*(idxD + r * Ns)] = v[2*r];
		b[2*(idxD + r * Ns)+1] = v[2*r+1];
	}
	

	
}


void main(int argc, char** argv)
{
	int  N  = pow(2,atoi(argv[1]));

	double * a = (double *) malloc(sizeof(double)* N * N* 2);
	double * b = (double *) malloc(sizeof(double)* N * N*2);
	double * c = (double *) malloc(sizeof(double)* N * N*2);
	double p = 2*M_PI ;	
	for (int i =0; i< N * N; i++)
	{
		a[2*i] = sin((p*i)/N);
		a[2*i+1] = 0;
		b[2*i] = 0;
		b[2*i+1] = 0;
	}

	cl_context ctx;
	cl_command_queue queue;
	create_context_on("Advanced Micro Devices","AMD",0,&ctx,&queue,0);

	cl_int status;
	cl_mem buf_a = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 
	sizeof(double) * N * N * 2 , 0, &status);
	CHECK_CL_ERROR(status, "clCreateBuffer");

	cl_mem buf_b = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 
	sizeof(double) * N *N * 2 , 0, &status);
	CHECK_CL_ERROR(status, "clCreateBuffer");
	
	cl_mem buf_c = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 
	sizeof(double) * N *N * 2 , 0, &status);
	CHECK_CL_ERROR(status, "clCreateBuffer");

	cl_mem buf_d = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 
	sizeof(double) * N *N * 2 , 0, &status);
	CHECK_CL_ERROR(status, "clCreateBuffer");
	cl_mem buf_e = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 
	sizeof(double) * N *N * 2 , 0, &status);
	CHECK_CL_ERROR(status, "clCreateBuffer");

	CALL_CL_GUARDED(clEnqueueWriteBuffer, (
	queue, buf_a, /*blocking*/ CL_TRUE, /*offset*/ 0,
	sizeof(double) * N * N * 2, a,
	0, NULL, NULL));

	CALL_CL_GUARDED(clEnqueueWriteBuffer, (
	queue, buf_b, /*blocking*/ CL_TRUE, /*offset*/ 0,
	sizeof(double) * N * N * 2, b,
	0, NULL, NULL));

	CALL_CL_GUARDED(clEnqueueWriteBuffer, (
	queue, buf_c, /*blocking*/ CL_TRUE, /*offset*/ 0,
	sizeof(double) * N * N * 2, c,
	0, NULL, NULL));

	CALL_CL_GUARDED(clEnqueueWriteBuffer, (
	queue, buf_d, /*blocking*/ CL_TRUE, /*offset*/ 0,
	sizeof(double) * N * N * 2, c,
	0, NULL, NULL));

	char *knl_text = read_file("fft1D.cl");
	cl_kernel fft1D = kernel_from_string(ctx, knl_text, "fft1D", NULL);
	free(knl_text);
	
	knl_text = read_file("transpose-soln.cl");
	cl_kernel mat_trans = kernel_from_string(ctx, knl_text, "transpose", NULL);
	free(knl_text);

	//fft_1D(buf_a,buf_b,buf_c, N, fft1D,queue,1,0);
	printf("ok\n");
	int Ns =1 ;
	int direction = 1;

	timestamp_type time1, time2;
	get_timestamp(&time1);

	fft2D(buf_a,buf_b,buf_c,buf_d,N,fft1D,mat_trans,queue, direction);

	CALL_CL_GUARDED(clFinish, (queue));

	get_timestamp(&time2);
	double elapsed = timestamp_diff_in_seconds(time1,time2);
	printf("on gpu %f s\n", elapsed);
	direction = -1;
	fft2D(buf_b,buf_c,buf_d,buf_e,N,fft1D,mat_trans,queue, direction);
	

	
	CALL_CL_GUARDED(clEnqueueReadBuffer, (
        	queue, buf_c, /*blocking*/ CL_TRUE, /*offset*/ 0,
       		2*N*N * sizeof(double), c,
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
		printf("%f + i*",c[2*i]/N/N);		
		printf("%f\n",c[2*i+1]/N/N);
	}

	/*printf("\n\n\non CPU!!!!\n\n\n");	

	
	for( Ns = 1;Ns < N; Ns *= 2 )
	{
		for (int j = 0; j<N/2; j++)
		{
			fftiteration(j,N,Ns,a,b);
		}
		double * c;
		c = a ;
		a = b;
		b = c;
		printf("ok\n");

	}*/


	




	
	CALL_CL_GUARDED(clReleaseMemObject, (buf_a));
	CALL_CL_GUARDED(clReleaseMemObject, (buf_b));
	
	CALL_CL_GUARDED(clReleaseKernel, (fft1D));
	CALL_CL_GUARDED(clReleaseCommandQueue, (queue));
	CALL_CL_GUARDED(clReleaseContext, (ctx));

}


