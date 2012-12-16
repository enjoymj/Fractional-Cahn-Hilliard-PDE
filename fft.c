#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include "cl-helper.h"
#include<CL/cl.h>

#ifndef PI
#define PI 3.14156265358979323846
#endif

#define Q 1
#define Nlaps2 2
#define Shar 3
#define Nlaps 4
#define X 5
#define Y 6


void mat__trans(cl_mem a, cl_mem b, int N, cl_kernel mat_trans, cl_command_queue queue,int option, float epsilon,float k,float s)
{

	SET_7_KERNEL_ARGS(mat_trans, a, b, N, option,epsilon,k,s);

	size_t ldim[] = { 16, 16 };
	size_t gdim[] = { N, N };
	CALL_CL_GUARDED(clEnqueueNDRangeKernel,
	(queue, mat_trans,
	/*dimensions*/ 2, NULL, gdim, ldim,
	0, NULL, NULL));


}



void fft_1D(cl_mem a,cl_mem b,cl_mem c, int N, cl_kernel init, cl_kernel knl,cl_command_queue queue,int direction,int offset_line)
{
	//handle complex-to-complex fft, accutal size = 2 * N

	//size_t ldim[] = { 128 };
	//size_t gdim[] = { (N /ldim[0])/2};
	int Ns = 1;
	int y =0;
	SET_7_KERNEL_ARGS(init, a, b, N, Ns,direction,offset_line,y);


	size_t ldim[] = { 1 };
	size_t gdim[] = { N/4 };

	CALL_CL_GUARDED(clEnqueueNDRangeKernel,
			(queue, init,
			 1, NULL, gdim, ldim,
			0, NULL, NULL));
	
	for(Ns=4; Ns<N; Ns<<=2)
	{



			SET_6_KERNEL_ARGS(knl, b, c, N, Ns,direction,offset_line);
			size_t ldim[] = { 1 };
			size_t gdim[] = { N/4 };
			
			CALL_CL_GUARDED(clEnqueueNDRangeKernel,
					(queue, knl,
					 1, NULL, gdim, ldim,
					0, NULL, NULL));
			clEnqueueCopyBuffer(queue,c,b,
					offset_line*N*2*sizeof(float),
					offset_line*N*2*sizeof(float),
					sizeof(float)*N*2,0,NULL,NULL);
			//VecCopy(c,b,N,offset_line,vec_copy,queue);
			
			
			
		  
	}
	
}

void fft_1D_w_orig(cl_mem a,cl_mem b,cl_mem c,int N, cl_kernel init_w, cl_kernel knl,cl_command_queue queue,int direction,int offset_line)
{
	//handle complex-to-complex fft, accutal size = 2 * N

	//size_t ldim[] = { 128 };
	//size_t gdim[] = { (N /ldim[0])/2};
	int Ns = 1;
	int option = 1;
	SET_7_KERNEL_ARGS(init_w, a, b, N, Ns,direction,offset_line,option);


	size_t ldim[] = { 1 };
	size_t gdim[] = { N/4 };

	CALL_CL_GUARDED(clEnqueueNDRangeKernel,
			(queue, init_w,
			 1, NULL, gdim, ldim,
			0, NULL, NULL));
	//CALL_CL_GUARDED(clFinish, (queue));
	for(Ns=4; Ns<N; Ns<<=2)
	{



			SET_6_KERNEL_ARGS(knl, b, c, N, Ns,direction,offset_line);
			size_t ldim[] = { 1 };
			size_t gdim[] = { N/4 };
			
			CALL_CL_GUARDED(clEnqueueNDRangeKernel,
					(queue, knl,
					 1, NULL, gdim, ldim,
					0, NULL, NULL));
			//CALL_CL_GUARDED(clFinish, (queue));
			clEnqueueCopyBuffer(queue,c,b,
					offset_line*N*2*sizeof(float),
					offset_line*N*2*sizeof(float),
					sizeof(float)*N*2,0,NULL,NULL);
			//VecCopy(c,b,N,offset_line,vec_copy,queue);
			
			//CALL_CL_GUARDED(clFinish, (queue));
			
		  
	}
	
}


void fft_1D_w(cl_mem a,cl_mem b,cl_mem c,cl_mem d, int N, cl_kernel init_w, cl_kernel knl,cl_command_queue queue,int direction,int offset_line)
{
	//handle complex-to-complex fft, accutal size = 2 * N

	//size_t ldim[] = { 128 };
	//size_t gdim[] = { (N /ldim[0])/2};
	int Ns = 1;
	//int option = 0;
	SET_7_KERNEL_ARGS(init_w, a, b,c, N, Ns,direction,offset_line);


	size_t ldim[] = { 1 };
	size_t gdim[] = { N/4 };

	CALL_CL_GUARDED(clEnqueueNDRangeKernel,
			(queue, init_w,
			 1, NULL, gdim, ldim,
			0, NULL, NULL));
	
	for(Ns=4; Ns<N; Ns<<=2)
	{



			SET_6_KERNEL_ARGS(knl, c, d, N, Ns,direction,offset_line);
			size_t ldim[] = { 1 };
			size_t gdim[] = { N/4 };
			
			CALL_CL_GUARDED(clEnqueueNDRangeKernel,
					(queue, knl,
					 1, NULL, gdim, ldim,
					0, NULL, NULL));
			clEnqueueCopyBuffer(queue,d,c,
					offset_line*N*2*sizeof(float),
					offset_line*N*2*sizeof(float),
					sizeof(float)*N*2,0,NULL,NULL);
			//VecCopy(c,b,N,offset_line,vec_copy,queue);
			
			
			
		  
	}
	
}
/* implementation of transpose-split method for 2D FFT*/


void fft2D(cl_mem a, cl_mem c, cl_mem b,cl_mem d, int N, cl_kernel fft_init,
		cl_kernel fft1D,cl_kernel mat_trans, cl_command_queue queue,int direction)
{
	

	for(int j= 0;j<N;j++)
	{
		fft_1D(a,b,c,N,fft_init,fft1D,queue,direction,j);
	}
	//CALL_CL_GUARDED(clFinish, (queue));
	//printf("1D fine \n");

	mat__trans(b,c,N,mat_trans,queue,0,1,1,1);

	//CALL_CL_GUARDED(clFinish, (queue));
	for(int j= 0;j<N;j++)
	{
		fft_1D(c,b,d,N,fft_init,fft1D,queue,direction,j);
	}
	//CALL_CL_GUARDED(clFinish, (queue));
	if(direction == 1)
		mat__trans(b,c,N,mat_trans,queue,0,1,1,1);
	else 
		mat__trans(b,c,N,mat_trans,queue,-1,1,1,1);
	
}



void fft_d_q(cl_mem a,cl_mem c,cl_mem b,cl_mem d, int N,float epsilon,float k,float s, cl_kernel fft_init,
		cl_kernel fft1D,cl_kernel mat_trans,cl_command_queue queue)
{
	for(int j= 0;j<N;j++)
	{
		fft_1D(a,b,c,N,fft_init,fft1D,queue,1,j);
	}
	//CALL_CL_GUARDED(clFinish, (queue));
	//printf("1D fine \n");

	mat__trans(b,c,N,mat_trans,queue,0,0,0,0);

	//CALL_CL_GUARDED(clFinish, (queue));
	for(int j= 0;j<N;j++)
	{
		fft_1D(c,b,d,N,fft_init,fft1D,queue,1,j);
	}
	//CALL_CL_GUARDED(clFinish, (queue));

	mat__trans(b,c,N,mat_trans,queue,Q,epsilon,k,s);
}


void fft_d_nlaps2(cl_mem a,cl_mem c,cl_mem b,cl_mem d,int N,float epsilon,float k,float s, cl_kernel fft_init,
		cl_kernel fft1D,cl_kernel mat_trans,cl_command_queue queue)
{
	for(int j= 0;j<N;j++)
	{
		fft_1D(a,b,c,N,fft_init,fft1D,queue,1,j);
	}
	//CALL_CL_GUARDED(clFinish, (queue));
	//printf("1D fine \n");

	mat__trans(b,c,N,mat_trans,queue,0,0,0,0);

	//CALL_CL_GUARDED(clFinish, (queue));
	for(int j= 0;j<N;j++)
	{
		fft_1D(c,b,d,N,fft_init,fft1D,queue,1,j);
	}
	//CALL_CL_GUARDED(clFinish, (queue));

	mat__trans(b,c,N,mat_trans,queue,Nlaps2,epsilon,k,s);
}


void fft_shar(cl_mem a ,cl_mem c ,cl_mem b,cl_mem d,int N,float epsilon,float k,float s, cl_kernel fft_init,
		cl_kernel fft1D,cl_kernel mat_trans,cl_command_queue queue)
{
	for(int j= 0;j<N;j++)
	{
		fft_1D(a,b,c,N,fft_init,fft1D,queue,1,j);
	}
	CALL_CL_GUARDED(clFinish, (queue));
	//printf("1D fine \n");

	mat__trans(b,c,N,mat_trans,queue,0,0,0,0);

	CALL_CL_GUARDED(clFinish, (queue));
	//printf("1D fine \n");
	for(int j= 0;j<N;j++)
	{
		fft_1D(c,b,d,N,fft_init,fft1D,queue,1,j);
	}
	CALL_CL_GUARDED(clFinish, (queue));

	mat__trans(b,c,N,mat_trans,queue,Shar,epsilon,k,s);

}


// fft(u.^3-u) .*nlap_s
void fft_w_orig(cl_mem a, cl_mem c, cl_mem b,cl_mem d, int N,float epsilon,float k,float s, cl_kernel fft_init,
		cl_kernel fft1D,cl_kernel mat_trans,cl_command_queue queue)
{

	for(int j= 0;j<N;j++)
	{
		fft_1D_w_orig(a,b,c,N,fft_init,fft1D,queue,1,j);
	}
	//CALL_CL_GUARDED(clFinish, (queue));
	//printf("1D fine \n");

	mat__trans(b,c,N,mat_trans,queue,0,0,0,0);

	//CALL_CL_GUARDED(clFinish, (queue));
	//printf("trans fine \n");

	for(int j= 0;j<N;j++)
	{
		fft_1D(c,b,d,N,fft_init,fft1D,queue,1,j);
	}
	//CALL_CL_GUARDED(clFinish, (queue));
	//printf("trans fine \n");
	mat__trans(b,c,N,mat_trans,queue,Nlaps,epsilon,k,s);
}

//fft((3*u.^2 -1) .* pk).*nlap_s
void fft_w(cl_mem a, cl_mem b, cl_mem c,/*result */cl_mem d,int N,float epsilon,float k,float s, 
		cl_kernel fft_init_w,cl_kernel fft_init,
		cl_kernel fft1D,cl_kernel mat_trans,cl_command_queue queue)
{
	for(int j= 0;j<N;j++)
	{
		fft_1D_w(a,b,c,d,N,fft_init_w,fft1D,queue,1,j);
	}
	//CALL_CL_GUARDED(clFinish, (queue));
	//printf("1D fine \n");
	


	mat__trans(c,d,N,mat_trans,queue,0,0,0,0);

	//CALL_CL_GUARDED(clFinish, (queue));
	for(int j= 0;j<N;j++)
	{
		fft_1D(d,c,d,N,fft_init,fft1D,queue,1,j);
	}
	//CALL_CL_GUARDED(clFinish, (queue));

	mat__trans(c,d,N,mat_trans,queue,Nlaps,epsilon,k,s);
}



void fft_d_x(cl_mem a,cl_mem c,cl_mem b,cl_mem d,int N,float epsilon,float k,float s, cl_kernel fft_init,
		cl_kernel fft1D,cl_kernel mat_trans,cl_command_queue queue)
{
	for(int j= 0;j<N;j++)
	{
		fft_1D(a,b,c,N,fft_init,fft1D,queue,1,j);
	}
	CALL_CL_GUARDED(clFinish, (queue));
	//printf("1D fine \n");

	mat__trans(b,c,N,mat_trans,queue,0,0,0,0);

	CALL_CL_GUARDED(clFinish, (queue));
	//printf("1D fine \n");
	for(int j= 0;j<N;j++)
	{
		fft_1D(c,b,d,N,fft_init,fft1D,queue,1,j);
	}
	CALL_CL_GUARDED(clFinish, (queue));

	mat__trans(b,c,N,mat_trans,queue,X,epsilon,k,s);
}

void fft_d_y(cl_mem a,cl_mem c,cl_mem b,cl_mem d,int N,float epsilon,float k,float s, cl_kernel fft_init,
		cl_kernel fft1D,cl_kernel mat_trans,cl_command_queue queue)
{
	for(int j= 0;j<N;j++)
	{
		fft_1D(a,b,c,N,fft_init,fft1D,queue,1,j);
	}
	CALL_CL_GUARDED(clFinish, (queue));
	//printf("1D fine \n");

	mat__trans(b,c,N,mat_trans,queue,0,0,0,0);

	CALL_CL_GUARDED(clFinish, (queue));
	//printf("1D fine \n");
	for(int j= 0;j<N;j++)
	{
		fft_1D(c,b,d,N,fft_init,fft1D,queue,1,j);
	}
	CALL_CL_GUARDED(clFinish, (queue));

	mat__trans(b,c,N,mat_trans,queue,Y,epsilon,k,s);
}

















