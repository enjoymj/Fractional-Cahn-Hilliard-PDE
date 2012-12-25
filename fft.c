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

#include"ch.h"

void mat__trans(cl_mem a, cl_mem b, int N, cl_kernel mat_trans, cl_command_queue queue,int option, float epsilon,float k,float s)
{

	cl_long offset = 0;
	SET_8_KERNEL_ARGS(mat_trans, a, b, N, option,epsilon,k,s,offset);

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


void fft_1D_new(cl_mem a,cl_mem b,cl_mem c, int N, cl_kernel init, cl_kernel interm, cl_kernel knl,cl_command_queue queue,int direction,int offset_line)
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

	if(N >= 4)
	{
	Ns = 4;

	SET_6_KERNEL_ARGS(interm, b, c, N, Ns,direction,offset_line);
	size_t ldim[] = { 16 };
	size_t gdim[] = { N/4 };
	
	CALL_CL_GUARDED(clEnqueueNDRangeKernel,
			(queue, interm,
			 1, NULL, gdim, ldim,
			0, NULL, NULL));
	clEnqueueCopyBuffer(queue,c,b,
			offset_line*N*2*sizeof(float),
			offset_line*N*2*sizeof(float),
			sizeof(float)*N*2,0,NULL,NULL);
	}
	if(N>=16)
	{
		Ns = 16;

		SET_6_KERNEL_ARGS(interm, b, c, N, Ns,direction,offset_line);
		size_t ldim[] = { 16 };
		size_t gdim[] = { N/4 };

		CALL_CL_GUARDED(clEnqueueNDRangeKernel,
		(queue, interm,
		 1, NULL, gdim, ldim,
		0, NULL, NULL));
		clEnqueueCopyBuffer(queue,c,b,
		offset_line*N*2*sizeof(float),
		offset_line*N*2*sizeof(float),
		sizeof(float)*N*2,0,NULL,NULL);
	}
	if(N >=64) 
	for(Ns=64; Ns<N; Ns<<=2)
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



void fft_d_q(cl_mem a,cl_mem c,cl_mem b,cl_mem d, int N,float epsilon,float k,float s, cl_kernel init_big,
		cl_kernel clean,cl_kernel mat_trans,cl_kernel mat_trans_3D,cl_command_queue queue)
{
	fft2D_transpose(a, c, b,d,N, init_big,
		clean,mat_trans, mat_trans_3D, queue,1,0);

	mat__trans(b,c,N,mat_trans,queue,Q,epsilon,k,s);
}


void fft_d_nlaps2(cl_mem a,cl_mem c,cl_mem b,cl_mem d,int N,float epsilon,float k,float s, cl_kernel init_big,
		cl_kernel clean,cl_kernel mat_trans,cl_kernel mat_trans_3D,cl_command_queue queue)
{
	fft2D_transpose(a, c, b,d,N, init_big,
		clean,mat_trans, mat_trans_3D, queue,1,0);

	mat__trans(b,c,N,mat_trans,queue,Nlaps2,epsilon,k,s);
}


void fft_shar(cl_mem a ,cl_mem c ,cl_mem b,cl_mem d,int N,float epsilon,float k,float s, cl_kernel init_big,
		cl_kernel clean,cl_kernel mat_trans,cl_kernel mat_trans_3D,cl_command_queue queue)
{

	fft2D_transpose(a, c, b,d,N, init_big,
		clean,mat_trans, mat_trans_3D, queue,1,0);
	mat__trans(b,c,N,mat_trans,queue,Shar,epsilon,k,s);

}


// fft(u.^3-u) .*nlap_s
void fft_w_orig(cl_mem a, cl_mem c, cl_mem b,cl_mem d, int N,float epsilon,float k,float s, cl_kernel init_big,
		cl_kernel clean,cl_kernel mat_trans,cl_kernel mat_trans_3D, cl_command_queue queue)
{

	fft2D_transpose(a, c, b,d,N, init_big,
		clean,mat_trans, mat_trans_3D, queue,1,1);


	mat__trans(b,c,N,mat_trans,queue,Nlaps,epsilon,k,s);
}

//fft((3*u.^2 -1) .* pk).*nlap_s
void fft_w(cl_mem a, cl_mem b, cl_mem c,/*result */cl_mem d,int N,float epsilon,float k,float s, 
		cl_kernel init_big,cl_kernel init_w,cl_kernel clean,
		cl_kernel mat_trans,cl_kernel mat_trans_3D,cl_command_queue queue)
{


		int direction = 1;
			int Ns = 1;
			int y =0;
		int offset_line = 0;
		SET_8_KERNEL_ARGS(init_w, a, b,c, N, Ns,direction,offset_line,y);


		size_t ldim[] = { 16 };
		size_t gdim[] = { N*N/4 };

		CALL_CL_GUARDED(clEnqueueNDRangeKernel,
				(queue, init_big,
				 1, NULL, gdim, ldim,
				0, NULL, NULL));
		
	if(N!=64)
		if(N == 1024)
		{
		
			int Ns =1;
			int y =0;			
			//cl_long offset = offset_line * N;
			SET_7_KERNEL_ARGS(clean, c, d, N, Ns,direction,offset_line,y);
			size_t ldim[]={ 4 };
			size_t gdim[] ={ N*N/4 };
			CALL_CL_GUARDED(clEnqueueNDRangeKernel,
				(queue, clean,
				 1, NULL, gdim, ldim,
				0, NULL, NULL));

		

			
			int option =0;
			float k =0;
			int n = 16;			
						
			SET_8_KERNEL_ARGS(mat_trans_3D, d, c, n, option,k,k,k,N);

				size_t ldim2[] = { 16, 16 ,1};
				size_t gdim2[] = { 16, 64 ,N};

			CALL_CL_GUARDED(clEnqueueNDRangeKernel,
				(queue, mat_trans_3D,
				3, NULL, gdim2, ldim2,
				0, NULL, NULL));

		}
		else if(N ==256)
		{

			int Ns =1;
			int y =0;			
			offset_line =0;
			SET_7_KERNEL_ARGS(clean, c, d, N, Ns,direction,offset_line,y);
			size_t ldim[] ={4};
			size_t gdim[] ={N*N/4};

			CALL_CL_GUARDED(clEnqueueNDRangeKernel,
				(queue, clean,
				 1, NULL, gdim, ldim,
				0, NULL, NULL));


							
			int option =0;
			float k =0;
			int n = 4;
						
			SET_8_KERNEL_ARGS(mat_trans_3D, d, c, n, option,k,k,k,N);

				size_t ldim2[] = { 4, 4 ,1};
				size_t gdim2[] = { 4, 64, N };

			CALL_CL_GUARDED(clEnqueueNDRangeKernel,
				(queue, mat_trans_3D,
				3, NULL, gdim2, ldim2,
				0, NULL, NULL));

			
		}
		
		else
		{
			printf("FFT not implemented for this size!!!\n");

			return;
		}	
	
	//CALL_CL_GUARDED(clFinish, (queue));
	//printf("1D fine \n");

	mat__trans(c,d,N,mat_trans,queue,0,1,1,1);

	//CALL_CL_GUARDED(clFinish, (queue));
/*	for(int j= 0;j<N;j++)
	{
		//fft_1D(c,b,d,N,fft_init,fft1D,queue,direction,j);
		fft_1D_big(c, b,d,N, init_big, clean,mat_trans,queue,direction,j);
	}
*/

		Ns =1;
		SET_7_KERNEL_ARGS(init_big, d, c, N, Ns,direction,offset_line,y);



		CALL_CL_GUARDED(clEnqueueNDRangeKernel,
				(queue, init_big,
				 1, NULL, gdim, ldim,
				0, NULL, NULL));

		
	if (N !=64 )
		

		if( N == 256 || N == 1024)
		{
			int Ns =1;
			int y = 0;			
			int offset_line = 0;
			SET_7_KERNEL_ARGS(clean, c, d, N, Ns,direction,offset_line,y);
			size_t ldim[] = { 4 };
			size_t gdim[] = { N*N/4 };

			CALL_CL_GUARDED(clEnqueueNDRangeKernel,
				(queue, clean,
				 1, NULL, gdim, ldim,
				0, NULL, NULL));

			if(N == 1024)
			{
			
			int option =0;
			float k =0;
			int n = 16;			
						
			SET_8_KERNEL_ARGS(mat_trans_3D, d, c, n, option,k,k,k,N);

				size_t ldim2[] = { 16, 16 ,1};
				size_t gdim2[] = { 16, 64 ,N};

			CALL_CL_GUARDED(clEnqueueNDRangeKernel,
				(queue, mat_trans_3D,
				3, NULL, gdim2, ldim2,
				0, NULL, NULL));
			
			}
			else if(N ==256)
			{
		
			int option =0;
			float k =0;
			int n = 4;
						
			SET_8_KERNEL_ARGS(mat_trans_3D, d, c, n, option,k,k,k,N);

				size_t ldim2[] = { 4, 4 ,1};
				size_t gdim2[] = { 4, 64, N };

			CALL_CL_GUARDED(clEnqueueNDRangeKernel,
				(queue, mat_trans_3D,
				3, NULL, gdim2, ldim2,
				0, NULL, NULL));

			
			}	
		
		}

		else
		{
			printf("FFT not implemented for this size!!!\n");

			return;
		}	

	mat__trans(c,d,N,mat_trans,queue,Nlaps,epsilon,k,s);
}




void fft_d_x(cl_mem a,cl_mem c,cl_mem b,cl_mem d,int N,float epsilon,float k,float s, cl_kernel init_big,
		cl_kernel clean,cl_kernel mat_trans,cl_kernel mat_trans_3D,cl_command_queue queue)
{
	fft2D_transpose(a, c, b,d,N, init_big,
		clean,mat_trans, mat_trans_3D, queue,1,0);

	mat__trans(b,c,N,mat_trans,queue,X,epsilon,k,s);
}


void fft_d_y(cl_mem a,cl_mem c,cl_mem b,cl_mem d,int N,float epsilon,float k,float s, cl_kernel init_big,
		cl_kernel clean,cl_kernel mat_trans,cl_kernel mat_trans_3D,cl_command_queue queue)
{
	fft2D_transpose(a, c, b,d,N, init_big,
		clean,mat_trans, mat_trans_3D, queue,1,0);

	mat__trans(b,c,N,mat_trans,queue,Y,epsilon,k,s);
}


void fft2D(cl_mem a, cl_mem c, cl_mem b,cl_mem d, int N, cl_kernel init_big,
		cl_kernel clean,cl_kernel mat_trans, cl_kernel mat_trans_3D, cl_command_queue queue,int direction)
{
	


		
			int Ns = 1;
			int y =0;
		int offset_line = 0;
		SET_7_KERNEL_ARGS(init_big, a, b, N, Ns,direction,offset_line,y);


		size_t ldim[] = { 16 };
		size_t gdim[] = { N*N/4 };

		CALL_CL_GUARDED(clEnqueueNDRangeKernel,
				(queue, init_big,
				 1, NULL, gdim, ldim,
				0, NULL, NULL));
		
	if(N!=64)
		if(N == 1024)
		{
		
			int Ns =1;
			int y =0;			
			//cl_long offset = offset_line * N;
			SET_7_KERNEL_ARGS(clean, b, c, N, Ns,direction,offset_line,y);
			size_t ldim[]={ 4 };
			size_t gdim[] ={ N*N/4 };
			CALL_CL_GUARDED(clEnqueueNDRangeKernel,
				(queue, clean,
				 1, NULL, gdim, ldim,
				0, NULL, NULL));

		

			
			int option =0;
			float k =0;
			int n = 16;			
						
			SET_8_KERNEL_ARGS(mat_trans_3D, c, b, n, option,k,k,k,N);

				size_t ldim2[] = { 16, 16 ,1};
				size_t gdim2[] = { 16, 64 ,N};

			CALL_CL_GUARDED(clEnqueueNDRangeKernel,
				(queue, mat_trans_3D,
				3, NULL, gdim2, ldim2,
				0, NULL, NULL));

		}
		else if(N ==256)
		{

			int Ns =1;
			int y =0;			
			offset_line =0;
			SET_7_KERNEL_ARGS(clean, b, c, N, Ns,direction,offset_line,y);
			size_t ldim[] ={4};
			size_t gdim[] ={N*N/4};

			CALL_CL_GUARDED(clEnqueueNDRangeKernel,
				(queue, clean,
				 1, NULL, gdim, ldim,
				0, NULL, NULL));


							
			int option =0;
			float k =0;
			int n = 4;
						
			SET_8_KERNEL_ARGS(mat_trans_3D, c, b, n, option,k,k,k,N);

				size_t ldim2[] = { 4, 4 ,1};
				size_t gdim2[] = { 4, 64, N };

			CALL_CL_GUARDED(clEnqueueNDRangeKernel,
				(queue, mat_trans_3D,
				3, NULL, gdim2, ldim2,
				0, NULL, NULL));

			
		}
		
		else
		{
			printf("FFT not implemented for this size!!!\n");

			return;
		}	
	
	//CALL_CL_GUARDED(clFinish, (queue));
	//printf("1D fine \n");

	mat__trans(b,c,N,mat_trans,queue,0,1,1,1);

	//CALL_CL_GUARDED(clFinish, (queue));
/*	for(int j= 0;j<N;j++)
	{
		//fft_1D(c,b,d,N,fft_init,fft1D,queue,direction,j);
		fft_1D_big(c, b,d,N, init_big, clean,mat_trans,queue,direction,j);
	}
*/

		Ns =1;
		SET_7_KERNEL_ARGS(init_big, c, b, N, Ns,direction,offset_line,y);



		CALL_CL_GUARDED(clEnqueueNDRangeKernel,
				(queue, init_big,
				 1, NULL, gdim, ldim,
				0, NULL, NULL));

		
	if (N !=64 )
		

		if( N == 256 || N == 1024)
		{
			int Ns =1;
			int y = 0;			
			int offset_line = 0;
			SET_7_KERNEL_ARGS(clean, b, d, N, Ns,direction,offset_line,y);
			size_t ldim[] = { 4 };
			size_t gdim[] = { N*N/4 };

			CALL_CL_GUARDED(clEnqueueNDRangeKernel,
				(queue, clean,
				 1, NULL, gdim, ldim,
				0, NULL, NULL));

			if(N == 1024)
			{
			
			int option =0;
			float k =0;
			int n = 16;			
						
			SET_8_KERNEL_ARGS(mat_trans_3D, d, b, n, option,k,k,k,N);

				size_t ldim2[] = { 16, 16 ,1};
				size_t gdim2[] = { 16, 64 ,N};

			CALL_CL_GUARDED(clEnqueueNDRangeKernel,
				(queue, mat_trans_3D,
				3, NULL, gdim2, ldim2,
				0, NULL, NULL));
			
			}
			else if(N ==256)
			{
		
			int option =0;
			float k =0;
			int n = 4;
						
			SET_8_KERNEL_ARGS(mat_trans_3D, d, b, n, option,k,k,k,N);

				size_t ldim2[] = { 4, 4 ,1};
				size_t gdim2[] = { 4, 64, N };

			CALL_CL_GUARDED(clEnqueueNDRangeKernel,
				(queue, mat_trans_3D,
				3, NULL, gdim2, ldim2,
				0, NULL, NULL));

			
			}	
		
		}

		else
		{
			printf("FFT not implemented for this size!!!\n");

			return;
		}	
	

	//CALL_CL_GUARDED(clFinish, (queue));
	if(direction == 1)
		mat__trans(b,c,N,mat_trans,queue,0,1,1,1);
	else 
		mat__trans(b,c,N,mat_trans,queue,-1,1,1,1);
	
}



void fft2D_transpose(cl_mem a, cl_mem c, cl_mem b,cl_mem d, int N, cl_kernel init_big,
		cl_kernel clean,cl_kernel mat_trans, cl_kernel mat_trans_3D, cl_command_queue queue,int direction,int y)
{
	

		
		int Ns = 1;
		int offset_line = 0;
		SET_7_KERNEL_ARGS(init_big, a, b, N, Ns,direction,offset_line,y);


		size_t ldim[] = { 16 };
		size_t gdim[] = { N*N/4 };

		CALL_CL_GUARDED(clEnqueueNDRangeKernel,
				(queue, init_big,
				 1, NULL, gdim, ldim,
				0, NULL, NULL));
		
	if(N!=64)
		if(N == 1024)
		{
		
			int Ns =1;
			int y =0;			
			//cl_long offset = offset_line * N;
			SET_7_KERNEL_ARGS(clean, b, c, N, Ns,direction,offset_line,y);
			size_t ldim[]={ 4 };
			size_t gdim[] ={ N*N/4 };
			CALL_CL_GUARDED(clEnqueueNDRangeKernel,
				(queue, clean,
				 1, NULL, gdim, ldim,
				0, NULL, NULL));

		

			
			int option =0;
			float k =0;
			int n = 16;			
						
			SET_8_KERNEL_ARGS(mat_trans_3D, c, b, n, option,k,k,k,N);

				size_t ldim2[] = { 16, 16 ,1};
				size_t gdim2[] = { 16, 64 ,N};

			CALL_CL_GUARDED(clEnqueueNDRangeKernel,
				(queue, mat_trans_3D,
				3, NULL, gdim2, ldim2,
				0, NULL, NULL));

		}
		else if(N ==256)
		{

			int Ns =1;
			int y =0;			
			offset_line =0;
			SET_7_KERNEL_ARGS(clean, b, c, N, Ns,direction,offset_line,y);
			size_t ldim[] ={4};
			size_t gdim[] ={N*N/4};

			CALL_CL_GUARDED(clEnqueueNDRangeKernel,
				(queue, clean,
				 1, NULL, gdim, ldim,
				0, NULL, NULL));


							
			int option =0;
			float k =0;
			int n = 4;
						
			SET_8_KERNEL_ARGS(mat_trans_3D, c, b, n, option,k,k,k,N);

				size_t ldim2[] = { 4, 4 ,1};
				size_t gdim2[] = { 4, 64, N };

			CALL_CL_GUARDED(clEnqueueNDRangeKernel,
				(queue, mat_trans_3D,
				3, NULL, gdim2, ldim2,
				0, NULL, NULL));

			
		}
		
		else
		{
			printf("FFT not implemented for this size!!!\n");

			return;
		}	
	
	//CALL_CL_GUARDED(clFinish, (queue));
	//printf("1D fine \n");


	mat__trans(b,c,N,mat_trans,queue,0,1,1,1);

	//CALL_CL_GUARDED(clFinish, (queue));
/*	for(int j= 0;j<N;j++)
	{
		//fft_1D(c,b,d,N,fft_init,fft1D,queue,direction,j);
		fft_1D_big(c, b,d,N, init_big, clean,mat_trans,queue,direction,j);
	}
*/

		Ns =1;
		SET_7_KERNEL_ARGS(init_big, c, b, N, Ns,direction,offset_line,y);



		CALL_CL_GUARDED(clEnqueueNDRangeKernel,
				(queue, init_big,
				 1, NULL, gdim, ldim,
				0, NULL, NULL));

		
	if (N !=64 )
		

		if( N == 256 || N == 1024)
		{
			int Ns =1;
			int y = 0;			
			int offset_line = 0;
			SET_7_KERNEL_ARGS(clean, b, d, N, Ns,direction,offset_line,y);
			size_t ldim[] = { 4 };
			size_t gdim[] = { N*N/4 };

			CALL_CL_GUARDED(clEnqueueNDRangeKernel,
				(queue, clean,
				 1, NULL, gdim, ldim,
				0, NULL, NULL));

			if(N == 1024)
			{
			
			int option =0;
			float k =0;
			int n = 16;			
						
			SET_8_KERNEL_ARGS(mat_trans_3D, d, b, n, option,k,k,k,N);

				size_t ldim2[] = { 16, 16 ,1};
				size_t gdim2[] = { 16, 64 ,N};

			CALL_CL_GUARDED(clEnqueueNDRangeKernel,
				(queue, mat_trans_3D,
				3, NULL, gdim2, ldim2,
				0, NULL, NULL));
			
			}
			else if(N ==256)
			{
		
			int option =0;
			float k =0;
			int n = 4;
						
			SET_8_KERNEL_ARGS(mat_trans_3D, d, b, n, option,k,k,k,N);

				size_t ldim2[] = { 4, 4 ,1};
				size_t gdim2[] = { 4, 64, N };

			CALL_CL_GUARDED(clEnqueueNDRangeKernel,
				(queue, mat_trans_3D,
				3, NULL, gdim2, ldim2,
				0, NULL, NULL));

			
			}	
		
		}

		else
		{
			printf("FFT not implemented for this size!!!\n");

			return;
		}	
	

	
}










