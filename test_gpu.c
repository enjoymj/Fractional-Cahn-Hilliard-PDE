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


#define Q 1
#define Nlaps2 2
#define Shar 3
#define Nlaps 4
#define X 5
#define Y 6


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

void fft_1D_big(cl_mem a,cl_mem b,cl_mem c, int N, cl_kernel init_big, cl_kernel clean,cl_kernel mat_trans,cl_command_queue queue,int direction,int offset_line)
{
	//handle complex-to-complex fft, accutal size = 2 * N

	//size_t ldim[] = { 128 };
	//size_t gdim[] = { (N /ldim[0])/2};
	int Ns = 1;
	int y =0;
	SET_7_KERNEL_ARGS(init_big, a, b, N, Ns,direction,offset_line,y);


	size_t ldim[] = { 16 };
	size_t gdim[] = { N/4 };

	CALL_CL_GUARDED(clEnqueueNDRangeKernel,
			(queue, init_big,
			 1, NULL, gdim, ldim,
			0, NULL, NULL));
	if (N ==64 )
		return;
	else
	if( N == 256 || N == 1024)
	{
		cl_long offset = offset_line * N;
		SET_7_KERNEL_ARGS(clean, b, c, N, Ns,direction,offset_line,y);
		ldim[0] =4;

		CALL_CL_GUARDED(clEnqueueNDRangeKernel,
			(queue, clean,
			 1, NULL, gdim, ldim,
			0, NULL, NULL));
		if(N == 1024)
		{
			int option =0;
			float k =0;
			int n = 16;			
			SET_8_KERNEL_ARGS(mat_trans, c, b, n, option,k,k,k,offset);

			size_t ldim[] = { 16, 16 };
			size_t gdim[] = { 16, 64 };
			CALL_CL_GUARDED(clEnqueueNDRangeKernel,
				(queue, mat_trans,
				2, NULL, gdim, ldim,
				0, NULL, NULL));

		}
		else if(N ==256)
		{
			int option =0;
			float k =0;
			int n = 4;			
			SET_8_KERNEL_ARGS(mat_trans, c, b, n, option,k,k,k,offset);

			size_t ldim[] = { 4, 4 };
			size_t gdim[] = { 4, 64 };
			CALL_CL_GUARDED(clEnqueueNDRangeKernel,
				(queue, mat_trans,
				2, NULL, gdim, ldim,
				0, NULL, NULL));

		}	
		
	}

	else
	{
		printf("FFT not implemented for this size!!!\n");

		return;
	}	
}


void fft_1D_new(cl_mem a,cl_mem b,cl_mem c, int N, cl_kernel init, cl_kernel interm, cl_kernel knl, cl_command_queue queue,int direction,int offset_line)
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

void fft2D_new(cl_mem a, cl_mem c, cl_mem b,cl_mem d, int N, cl_kernel init,cl_kernel interm,
		cl_kernel fft1D,cl_kernel mat_trans, cl_command_queue queue,int direction)
{
#if 0	
		int Ns = 1;
		int y =0;
		int x =N*N;
		SET_7_KERNEL_ARGS(init, a, b, N, Ns,direction,y,y);


		size_t ldim[] = { 1 };
		size_t gdim[] = { N*N/4 };

		CALL_CL_GUARDED(clEnqueueNDRangeKernel,
			(queue, init,
			 1, NULL, gdim, ldim,
			0, NULL, NULL));
#endif

#if 1
	int Ns = 1;
	int stride = 64;
	for(int blk=0; blk<stride;blk++)
		for(int j= 0;j<N/stride;j++)
		{
			int offset = blk*N/stride +j;
		
		int y =0;
		SET_7_KERNEL_ARGS(init, a, b, N, Ns,direction,offset,y);


		size_t ldim[] = { 1 };
		size_t gdim[] = { N/4 };

		CALL_CL_GUARDED(clEnqueueNDRangeKernel,
			(queue, init,
			 1, NULL, gdim, ldim,
			0, NULL, NULL));
	}
#if 1
	for(int blk=0; blk<stride;blk++)
		for(int j= 0;j<N/stride;j++)
		{
			int offset = blk*N/stride +j;
		if(N >= 4)
		{
		Ns = 4;

		SET_6_KERNEL_ARGS(interm, b, c, N, Ns,direction,offset);
		size_t ldim[] = { 16 };
		size_t gdim[] = { N/4 };
		CALL_CL_GUARDED(clEnqueueNDRangeKernel,
		(queue, interm,
		 1, NULL, gdim, ldim,
		0, NULL, NULL));

		}
}
		
	
		clEnqueueCopyBuffer(queue,c,b,
			0,
			0,
			sizeof(float)*N*N*2,0,NULL,NULL);


	for(int blk=0; blk<stride;blk++)
		for(int j= 0;j<N/stride;j++)
		{
			int offset = blk*N/stride +j;
		if(N>=16)
		{
		Ns = 16;

		SET_6_KERNEL_ARGS(interm, b, c, N, Ns,direction,offset);
		size_t ldim[] = { 16 };
		size_t gdim[] = { N/4 };

		CALL_CL_GUARDED(clEnqueueNDRangeKernel,
		(queue, interm,
		1, NULL, gdim, ldim,
		0, NULL, NULL));

		}

}

		clEnqueueCopyBuffer(queue,c,b,
		0,
		0,
		sizeof(float)*N*N*2,0,NULL,NULL);

	if(N >=64) 

		
#endif
		for(Ns=64; Ns<N; Ns<<=2)
		{

	for(int blk=0; blk<stride;blk++)
		for(int j= 0;j<N/stride;j++)
		{
			int offset = blk*N/stride +j;

		SET_6_KERNEL_ARGS(fft1D, b, c, N, Ns,direction,offset);
		size_t ldim[] = { 1 };
		size_t gdim[] = { N/4 };

		CALL_CL_GUARDED(clEnqueueNDRangeKernel,
				(queue, fft1D,
				 1, NULL, gdim, ldim,
				0, NULL, NULL));

		//VecCopy(c,b,N,offset_line,vec_copy,queue);




		}

		clEnqueueCopyBuffer(queue,c,b,
				0,
				0,
				sizeof(float)*N*N*2,0,NULL,NULL);
		}

#endif
	//CALL_CL_GUARDED(clFinish, (queue));
	//printf("1D fine \n");

	mat__trans(b,c,N,mat_trans,queue,0,1,1,1);

	#if 0
	float test;
	CALL_CL_GUARDED(clFinish, (queue));
	CALL_CL_GUARDED(clEnqueueReadBuffer, (
        	queue, c, /*blocking*/ CL_TRUE, /*offset*/ 2*sizeof(float)*N,
       		sizeof(float), &test,
        	0, NULL, NULL));
	

	printf("test = %f\n",test);
	#endif



	//CALL_CL_GUARDED(clFinish, (queue));

#if 0
	for(int j= 0;j<N;j++)
	{
		fft_1D_new(c,b,d,N,init,interm,fft1D,queue,direction,j);
	}
#endif
#if 1
	Ns = 1;	
	

	for(int blk=0; blk<stride;blk++)
		for(int j= 0;j<N/stride;j++)
		{
			int offset = blk*N/stride +j;
		
		int y =0;
		SET_7_KERNEL_ARGS(init, c, b, N, Ns,direction,offset,y);


		size_t ldim[] = { 1 };
		size_t gdim[] = { N/4 };

		CALL_CL_GUARDED(clEnqueueNDRangeKernel,
			(queue, init,
			 1, NULL, gdim, ldim,
			0, NULL, NULL));
	}
#if 1
	for(int blk=0; blk<stride;blk++)
		for(int j= 0;j<N/stride;j++)
		{
			int offset = blk*N/stride +j;
		if(N >= 4)
		{
		Ns = 4;

		SET_6_KERNEL_ARGS(interm, b, d, N, Ns,direction,offset);
		size_t ldim[] = { 16 };
		size_t gdim[] = { N/4 };
		CALL_CL_GUARDED(clEnqueueNDRangeKernel,
		(queue, interm,
		 1, NULL, gdim, ldim,
		0, NULL, NULL));

		}
}
		
	
		clEnqueueCopyBuffer(queue,d,b,
			0,
			0,
			sizeof(float)*N*N*2,0,NULL,NULL);


	for(int blk=0; blk<stride;blk++)
		for(int j= 0;j<N/stride;j++)
		{
			int offset = blk*N/stride +j;
		if(N>=16)
		{
		Ns = 16;

		SET_6_KERNEL_ARGS(interm, b, d, N, Ns,direction,offset);
		size_t ldim[] = { 16 };
		size_t gdim[] = { N/4 };

		CALL_CL_GUARDED(clEnqueueNDRangeKernel,
		(queue, interm,
		1, NULL, gdim, ldim,
		0, NULL, NULL));

		}

}

		clEnqueueCopyBuffer(queue,d,b,
		0,
		0,
		sizeof(float)*N*N*2,0,NULL,NULL);

	if(N >=64) 

#endif		

		for(Ns=64; Ns<N; Ns<<=2)
		{

	for(int blk=0; blk<stride;blk++)
		for(int j= 0;j<N/stride;j++)
		{
			int offset = blk*N/stride +j;

		SET_6_KERNEL_ARGS(fft1D, b, d, N, Ns,direction,offset);
		size_t ldim[] = { 1 };
		size_t gdim[] = { N/4 };

		CALL_CL_GUARDED(clEnqueueNDRangeKernel,
				(queue, fft1D,
				 1, NULL, gdim, ldim,
				0, NULL, NULL));

		//VecCopy(c,b,N,offset_line,vec_copy,queue);




		}

		clEnqueueCopyBuffer(queue,d,b,
				0,
				0,
				sizeof(float)*N*N*2,0,NULL,NULL);
		}
#endif

	//CALL_CL_GUARDED(clFinish, (queue));
	if(direction == 1)
		mat__trans(b,c,N,mat_trans,queue,0,1,1,1);
	else 
		mat__trans(b,c,N,mat_trans,queue,-1,1,1,1);
	
}
#if 0
void fft2D_big(cl_mem a, cl_mem c, cl_mem b,cl_mem d, int N, cl_kernel init_big,
		cl_kernel clean,cl_kernel mat_trans, cl_command_queue queue,int direction)
{
	

	for(int j= 0;j<N;j++)
	{
		
		fft_1D_big(a, b,c,N, init_big, clean,mat_trans,queue,direction,j);
	}
	//CALL_CL_GUARDED(clFinish, (queue));
	//printf("1D fine \n");

	mat__trans(b,c,N,mat_trans,queue,0,1,1,1);

	//CALL_CL_GUARDED(clFinish, (queue));
	for(int j= 0;j<N;j++)
	{
		//fft_1D(c,b,d,N,fft_init,fft1D,queue,direction,j);
		fft_1D_big(c, b,d,N, init_big, clean,mat_trans,queue,direction,j);
	}
	//CALL_CL_GUARDED(clFinish, (queue));
	if(direction == 1)
		mat__trans(b,c,N,mat_trans,queue,0,1,1,1);
	else 
		mat__trans(b,c,N,mat_trans,queue,-1,1,1,1);
	
}
#endif

#if 1

void fft2D_big(cl_mem a, cl_mem c, cl_mem b,cl_mem d, int N, cl_kernel init_big,
		cl_kernel clean,cl_kernel mat_trans, cl_command_queue queue,int direction)
{
	

	for(int j= 0;j<N;j++)
	{
		int offset_line = j;
			int Ns = 1;
			int y =0;
		SET_7_KERNEL_ARGS(init_big, a, b, N, Ns,direction,offset_line,y);


		size_t ldim[] = { 16 };
		size_t gdim[] = { N/4 };

		CALL_CL_GUARDED(clEnqueueNDRangeKernel,
				(queue, init_big,
				 1, NULL, gdim, ldim,
				0, NULL, NULL));
	}
		
		if(N !=64)

		if(N == 1024)
		{
		for(int j= 0;j<N;j++)
		{
			int offset_line = j;			
			int Ns =1;
			int y =0;			
			cl_long offset = offset_line * N;
			SET_7_KERNEL_ARGS(clean, b, c, N, Ns,direction,offset_line,y);
			size_t ldim[]={ 4 };
			size_t gdim[] ={ N/4 };
			CALL_CL_GUARDED(clEnqueueNDRangeKernel,
				(queue, clean,
				 1, NULL, gdim, ldim,
				0, NULL, NULL));

		

			int option =0;
			float k =0;
			int n = 16;			
			//cl_long offset = 0;			
			SET_8_KERNEL_ARGS(mat_trans, c, b, n, option,k,k,k,offset);

				size_t ldim2[] = { 16, 16 };
				size_t gdim2[] = { 16, 64 };

			CALL_CL_GUARDED(clEnqueueNDRangeKernel,
				(queue, mat_trans,
				2, NULL, gdim2, ldim2,
				0, NULL, NULL));
		}
		}
		else if(N ==256)
		{
			for(int j= 0;j<N;j++)
		{
			int offset_line = j;	
			int Ns =1;
			int y =0;			
			cl_long offset = offset_line * N;
			SET_7_KERNEL_ARGS(clean, b, c, N, Ns,direction,offset_line,y);
			size_t ldim[] ={4};
			size_t gdim[] ={N/4};

			CALL_CL_GUARDED(clEnqueueNDRangeKernel,
				(queue, clean,
				 1, NULL, gdim, ldim,
				0, NULL, NULL));				
			int option =0;
			float k =0;
			int n = 4;			
			SET_8_KERNEL_ARGS(mat_trans, c, b, n, option,k,k,k,offset);

				size_t ldim2[] = { 4, 4 };
				size_t gdim2[] = { 4, 64 };

			CALL_CL_GUARDED(clEnqueueNDRangeKernel,
				(queue, mat_trans,
				2, NULL, gdim2, ldim2,
				0, NULL, NULL));

		}	
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
for(int j= 0;j<N;j++)
	{
		int offset_line = j;
			int Ns = 1;
			int y =0;
		SET_7_KERNEL_ARGS(init_big, c, b, N, Ns,direction,offset_line,y);


		size_t ldim[] = { 16 };
		size_t gdim[] = { N/4 };

		CALL_CL_GUARDED(clEnqueueNDRangeKernel,
				(queue, init_big,
				 1, NULL, gdim, ldim,
				0, NULL, NULL));
}
		
	if(N !=64){
for(int j= 0;j<N;j++)
	{
		int offset_line = j;
		if( N == 256 || N == 1024)
		{
			int Ns =1;
			int y = 0;			
			cl_long offset = offset_line * N;
			SET_7_KERNEL_ARGS(clean, b, d, N, Ns,direction,offset_line,y);
			size_t ldim[] = { 4 };
			size_t gdim[] = { N/4 };

			CALL_CL_GUARDED(clEnqueueNDRangeKernel,
				(queue, clean,
				 1, NULL, gdim, ldim,
				0, NULL, NULL));
			if(N == 1024)
			{
				int option =0;
				float k =0;
				int n = 16;			
				SET_8_KERNEL_ARGS(mat_trans, d, b, n, option,k,k,k,offset);

				size_t ldim[] = { 16, 16 };
				size_t gdim[] = { 16, 64 };
				CALL_CL_GUARDED(clEnqueueNDRangeKernel,
					(queue, mat_trans,
					2, NULL, gdim, ldim,
					0, NULL, NULL));

			}
			else if(N ==256)
			{
				int option =0;
				float k =0;
				int n = 4;			
				SET_8_KERNEL_ARGS(mat_trans, d, b, n, option,k,k,k,offset);

				size_t ldim[] = { 4, 4 };
				size_t gdim[] = { 4, 64 };
				CALL_CL_GUARDED(clEnqueueNDRangeKernel,
					(queue, mat_trans,
					2, NULL, gdim, ldim,
					0, NULL, NULL));

			}	
		
		}

		else
		{
			printf("FFT not implemented for this size!!!\n");

			return;
		}	
	}
}
	//CALL_CL_GUARDED(clFinish, (queue));
	if(direction == 1)
		mat__trans(b,c,N,mat_trans,queue,0,1,1,1);
	else 
		mat__trans(b,c,N,mat_trans,queue,-1,1,1,1);
	
}

#endif
#if 1
void fft2D_big_new(cl_mem a, cl_mem c, cl_mem b,cl_mem d, int N, cl_kernel init_big,
		cl_kernel clean,cl_kernel mat_trans, cl_kernel mat_trans_3D, cl_command_queue queue,int direction)
{
	


		int offset_line = 0;
			int Ns = 1;
			int y =0;
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




#endif
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
	create_context_on("Advanced Micro Devices","Turks",0,&ctx,&queue,0);

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
	double elapsed ;

	CALL_CL_GUARDED(clFinish, (queue));
	get_timestamp(&time1);

//for(int s=0;s<100;s++)
	//fft_1D_big(buf_a,buf_b,buf_c,N,fft_big,fft_clean,mat_trans,queue,direction,0);
	//fft_1D_new(buf_a,buf_b,buf_c,N,fft_init,fft_interm, fft1D,queue,direction,0);
	//fft_1D(buf_a,buf_b,buf_c,N,fft_init, fft1D,queue,direction,0);
	//fft2D(buf_a,buf_b,buf_c,buf_d,N,fft_init,fft1D,mat_trans,queue, 1);
	//fft2D_new(buf_a,buf_b,buf_c,buf_d,N,fft_init,fft_interm,fft1D,mat_trans,queue, 1);
	//fft2D_big(buf_a,buf_b,buf_c,buf_d,N,fft_big,fft_clean,mat_trans,queue,direction);
	//fft2D_big_new(buf_a,buf_b,buf_c,buf_d,N,fft_2D,fft_2D_clean,
			//mat_trans,mat_trans_3D,queue,direction);
	//fft_w(buf_a,buf_b,buf_c,buf_d,buf_e,N,0.1,0,1,fft_init_w,fft_init,fft1D,mat_trans,queue);
#if 0
	frhs(buf_a,buf_b,buf_c,buf_d,buf_e,&param,fft1D_init,fft1D,mat_trans,
		 vec_add, queue);
#endif
#if 0	
	float E1 = energy(buf_a, buf_b, buf_c,buf_d, buf_e,buf_f,1e-4, 
				&param, fft_init,fft1D,mat_trans,reduct_eng,
				reduct,queue);
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

	
	get_timestamp(&time1);
	fft2D(buf_a,buf_b,buf_c,buf_d,N,fft_init,fft1D,mat_trans,queue, 1);
	CALL_CL_GUARDED(clFinish, (queue));
	get_timestamp(&time2);
	elapsed = timestamp_diff_in_seconds(time1,time2);
	printf("Navie 2D FFT of size %d * %d matrix  on gpu takes %f s\n", N,N,elapsed);
	printf("achieve %f GFLOPS \n",6*8*N*N*k/elapsed*1e-9);
	printf("---------------------------------------------\n");
	//printf("data access from global achieve %f GB/s\n",sizeof(float)*2*16*N*N/elapsed*1e-9);
	CALL_CL_GUARDED(clFinish, (queue));
	get_timestamp(&time1);
	fft2D_new(buf_a,buf_b,buf_c,buf_d,N,fft_init,fft_interm,fft1D,mat_trans,queue, 1);
	CALL_CL_GUARDED(clFinish, (queue));
	get_timestamp(&time2);
	elapsed = timestamp_diff_in_seconds(time1,time2);
	printf("local data exchange 2D FFT of size %d * %d matrix  on gpu takes %f s\n", N,N,elapsed);
	printf("achieve %f GFLOPS \n",6*8*N*N*k/elapsed*1e-9);
	printf("---------------------------------------------\n");


	CALL_CL_GUARDED(clFinish, (queue));
	get_timestamp(&time1);
	fft2D_big(buf_a,buf_b,buf_c,buf_d,N,fft_big,fft_clean,mat_trans,queue,direction);
	CALL_CL_GUARDED(clFinish, (queue));
	get_timestamp(&time2);
	elapsed = timestamp_diff_in_seconds(time1,time2);
	printf("Hierarchy 2D FFT of size %d * %d matrix  on gpu takes %f s\n", N,N,elapsed);
	printf("achieve %f GFLOPS \n",6*8*N*N*k/elapsed*1e-9);
	printf("---------------------------------------------\n");


	CALL_CL_GUARDED(clFinish, (queue));
	get_timestamp(&time1);
	fft2D_big_new(buf_a,buf_b,buf_c,buf_d,N,fft_2D,fft_2D_clean,
			mat_trans,mat_trans_3D,queue,direction);
	CALL_CL_GUARDED(clFinish, (queue));
	get_timestamp(&time2);
	elapsed = timestamp_diff_in_seconds(time1,time2);
	printf("Using 2D kernel 2D FFT of size %d * %d matrix  on gpu takes %f s\n", N,N,elapsed);
	printf("achieve %f GFLOPS \n",6*8*N*N*k/elapsed*1e-9);
	printf("---------------------------------------------\n");



	get_timestamp(&time1);






	direction = -1;
	//fft_1D(buf_b,buf_c,buf_d,N,fft_init, fft1D,queue,direction,0);
	fft2D(buf_b,buf_c,buf_d,buf_e,N,fft_init,fft1D,mat_trans,queue, direction);
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
	#if 0
	CALL_CL_GUARDED(clFinish, (queue));
	CALL_CL_GUARDED(clEnqueueReadBuffer, (
        	queue, buf_c, /*blocking*/ CL_TRUE, /*offset*/ 0,
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
	CALL_CL_GUARDED(clReleaseKernel, (fft_init));
	CALL_CL_GUARDED(clReleaseKernel, (vec_add));
	CALL_CL_GUARDED(clReleaseKernel, (reduct_mul));
	CALL_CL_GUARDED(clReleaseKernel, (reduct));
	CALL_CL_GUARDED(clReleaseKernel, (mat_trans));
	CALL_CL_GUARDED(clReleaseCommandQueue, (queue));
	CALL_CL_GUARDED(clReleaseContext, (ctx));

}


