// Code CH2D for H_s norm gradient flow of Modica-Mortola energy

// Created by Kangping Zhu, kangping@cims.nyu.edu
// Derived from Brian Wetton's algorithm and matlab code, wetton@math.ubc.ca

// Uses subroutines:
// unint:    defines intital data for the computation
// uplot:    outputs a contour plot of solutions
// frhs:     right hand side of the method of lines ODE 
// chstep:   calls ch1step, allows generalization to other times stepping
// ch1step:  calls chcg, Newton iterations for implicit substeps
// chcg:     preconditioned conjugate gradient iterations for ch1step
// bigE:     error estimation 
// energy:   calculates underlying energy of a solution
 
// Structures param and symbol contain information passed to subroutines
//
//----------
// User defined parameters
//
// set numerical, physical and run parameters


// take parameter s as the power for fractional laplacian 
// epsilon as the width of boundary layer. 
// it should at least be around 10*1/N



#include "cl-helper.h"
#include "timing.h"
#include<CL/cl.h>
#include "ppm.h"
#include<stdbool.h>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#ifndef PI
#define PI 3.14156265358979323846
#endif


#include"ch.h"






int main(int argc, char** argv)
{
	if(argc !=4)
	{
		fprintf(stderr, "need three arguments!\n");
    		abort();
	}
	
	float s = atof(argv[1]);
	float epsilon = atof(argv[2]);
        int   N = atoi(argv[3]);


	float finalT = 10;

	// Acceptable error per time step
	// Note that sigma is called delta in the paper
	float sigma = 1e-4;

	print_platforms_devices();

	cl_context ctx;
	cl_command_queue queue;
	create_context_on("Advanced Micro Devices","Turks",0,&ctx,&queue,0);
	

	printf("Context created! \n");

	struct parameter param;
	param.h = 2*PI/N;
	param.N = N;
	param.epsilon = epsilon;
 	param.maxCG = 100;
	param.maxN = 5;
	param.s = s;
	//Minimum and starting time step
	float mink = 1e-7;
	float startk = 1e-4;

	// Tolerances
	param.Ntol = 1e-4;
	param.cgtol = 1e-4;
	float ksafety = 0.8;
	float kfact = 1.3;
	float kfact2 = 1/1.3;
	float Nfact = 0.7;
	float CGfact = 0.7;
	// --------------------------------------------------------------------------
  	// allocate and initialize CPU memory
  	// --------------------------------------------------------------------------
  	float *uu = (float *) malloc(sizeof(float) * N * N * 2);
  	if (!uu) { perror("alloc uu"); abort(); }
  	float *uuu = (float *) malloc(sizeof(float) * N * N * 2);
  	if (!uuu) { perror("alloc uuu"); abort(); }
	srand(1);
	for(int i =0;i<N*N;i++)
	{	
		
		uu[2*i] =1;
		//printf("%f\n",uu[2*i]);
		uu[2*i+1] =0 ;
	}
	printf("Initialized data %f\n",uu[0]);
  	//float *u_hat = (float *) malloc(sizeof(float) * N * N * 2);
  	//if (!u_hat) { perror("alloc u_hat"); abort(); }
	//float *temp = (float *) malloc(sizeof(float) * N * N * 2);
  	//if (!temp) { perror("alloc w_d_hat"); abort(); }


	// --------------------------------------------------------------------------
	// allocate device memory
	// --------------------------------------------------------------------------
	cl_int status;
	cl_mem u = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 
	sizeof(float) * N * N * 2 , 0, &status);
	CHECK_CL_ERROR(status, "clCreateBuffer");

	cl_mem fu0 = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
	sizeof(float) * N * N * 2, 0, &status);
	CHECK_CL_ERROR(status, "clCreateBuffer");

	cl_mem u1 = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
	sizeof(float) * N * N * 2, 0, &status);
	CHECK_CL_ERROR(status, "clCreateBuffer");

	cl_mem fu1 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 
	sizeof(float) * N * N * 2 , 0, &status);
	CHECK_CL_ERROR(status, "clCreateBuffer");

	cl_mem rhs = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 
	sizeof(float) * N * N * 2 , 0, &status);
	CHECK_CL_ERROR(status, "clCreateBuffer");
	cl_mem temp = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 
	sizeof(float) * N * N * 2 , 0, &status);
	CHECK_CL_ERROR(status, "clCreateBuffer");

	cl_mem temp2 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 
	sizeof(float) * N * N * 2 , 0, &status);
	CHECK_CL_ERROR(status, "clCreateBuffer");
	cl_mem temp3 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 
	sizeof(float) * N * N * 2 , 0, &status);
	CHECK_CL_ERROR(status, "clCreateBuffer");
	cl_mem temp4 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 
	sizeof(float) * N * N * 2 , 0, &status);
	CHECK_CL_ERROR(status, "clCreateBuffer");
	cl_mem temp5 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 
	sizeof(float) * N * N * 2 , 0, &status);
	CHECK_CL_ERROR(status, "clCreateBuffer");
	cl_mem temp6 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 
	sizeof(float) * N * N * 2 , 0, &status);
	CHECK_CL_ERROR(status, "clCreateBuffer");


	cl_mem temp7 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 
	sizeof(float) * N * N * 2 , 0, &status);
	CHECK_CL_ERROR(status, "clCreateBuffer");
	cl_mem temp8 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 
	sizeof(float) * N * N * 2 , 0, &status);
	CHECK_CL_ERROR(status, "clCreateBuffer");
	cl_mem temp9 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 
	sizeof(float) * N * N * 2 , 0, &status);
	CHECK_CL_ERROR(status, "clCreateBuffer");

	printf("all buffer created! \n");
	// --------------------------------------------------------------------------
	// transfer to device
	// --------------------------------------------------------------------------
	CALL_CL_GUARDED(clEnqueueWriteBuffer, (
	queue, u, /*blocking*/ CL_TRUE, /*offset*/ 0,
	2*N*N * sizeof(float), uu,
	0, NULL, NULL));



	//--------------------------------
	//load kernels
	//--------------------------------


	char *knl_text = read_file("vec_add.cl");
	cl_kernel vec_add = kernel_from_string(ctx, knl_text, "sum", NULL);
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
	
	knl_text = read_file("transpose-soln-gpu.cl");
	cl_kernel mat_trans = kernel_from_string(ctx, knl_text, "transpose", NULL);
	free(knl_text);

	knl_text = read_file("radix-4-2D-modi.cl");
	cl_kernel fft_init_w = kernel_from_string(ctx, knl_text, "fft2D_big", NULL);
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


	//-----------------------------------
	// Run algorithm on device
	//-----------------------------------


	//frhs(u,fu0,temp,temp2,temp3,&param,&knl,&queue);

	CALL_CL_GUARDED(clFinish, (queue));
	frhs(u,fu0,temp,temp2,temp3,&param,fft_2D,fft_2D_clean,mat_trans,mat_trans_3D,
		 vec_add, queue);


	#if 0
	float test;
	CALL_CL_GUARDED(clFinish, (queue));
	CALL_CL_GUARDED(clEnqueueReadBuffer, (
        	queue, fu0, /*blocking*/ CL_TRUE, /*offset*/ 0,
       		sizeof(float), &test,
        	0, NULL, NULL));
	

		printf("test success and %f \n", test);		
	

	#endif
	
	//printf("frhs done\n");
	float E0 = energy(u, temp9, temp4,temp5, temp6,temp7,0, 
		&param, fft_2D,fft_2D_clean,mat_trans,mat_trans_3D,reduct_eng,
		reduct,queue); 
	float kk = startk;
	bool epic_fail = false;
	bool fail = false;
	bool accfail = false;
	float tt = 0;
	float Xi = 0;
	//float * temp_ptr = NULL;
	float kknew;
	
	float E1;
	while(epic_fail == false && tt < finalT)
	{
		//chstep(u,fu0,u1,2*N*N,&fail,kK,&param,&knl,&queue);
		CALL_CL_GUARDED(clFinish, (queue));
		printf("Next time stepping\n");
		chstep(u, fu0, u1, rhs, fu1, temp, temp2,temp3,
			temp4,temp5, temp6,temp7,temp8,temp9,
			N, &fail, kk, &param, 
			fft_2D,fft_2D_clean, fft_init_w,vec_add,
			vec_zero,mat_trans,mat_trans_3D,reduct,reduct_mul,
			reduct_init,resid,resid_init,queue);
		CALL_CL_GUARDED(clFinish, (queue));
#if 0 
	CALL_CL_GUARDED(clFinish, (queue));
	CALL_CL_GUARDED(clEnqueueReadBuffer, (
        	queue, u1, /*blocking*/ CL_TRUE, /*offset*/ 0,
       		2*N*N* sizeof(float), uu,
        	0, NULL, NULL));
	CALL_CL_GUARDED(clEnqueueReadBuffer, (
        	queue, u, /*blocking*/ CL_TRUE, /*offset*/ 0,
       		2*N*N* sizeof(float), uuu,
        	0, NULL, NULL));
	

	/*for(int i =0; i<  N; i++)
	{
		printf("a%f+ i*",a[2*i]);		
		printf("%f\n",a[2*i+1]);
	}*/
	int T = 10<N? 10:N ;
	for(int i =0; i<  T; i++)
	{
		printf("%f + i*",uu[2*i]);		
		printf("%f\t",uu[2*i+1]);
		printf("%f + i*",uuu[2*i]);		
		printf("%f\n",uuu[2*i+1]);
	}

	#endif 




		

		if(fail == false)
		{
			CALL_CL_GUARDED(clFinish, (queue));
			frhs(u1,fu1,temp,temp2,temp3,&param,fft_2D,fft_2D_clean,mat_trans,mat_trans_3D,
			 vec_add, queue);


		#if 0
		float test;
		CALL_CL_GUARDED(clFinish, (queue));
		CALL_CL_GUARDED(clEnqueueReadBuffer, (
		queue, u1, /*blocking*/ CL_TRUE, /*offset*/ 0,
		sizeof(float), &test,
		0, NULL, NULL));


		printf("in CH2d loop and u1 = %f \n",test);		


		#endif



			//E1 = energy(u1);
			E1 = energy(u1, temp9, temp4,temp5, temp6,temp7,kk, 
				&param, fft_2D,fft_2D_clean,mat_trans,mat_trans_3D,reduct_eng,
				reduct,queue); 
			CALL_CL_GUARDED(clFinish, (queue));
			printf("Energy now %f!!!!!!!!!!!!!!!\n",E1);
			//E1 =0;
			if (E1 > E0 * (1 +sigma) )
			{
				printf("Energy increase, failing time step\n");
				fail = true;
			}
		}
//CALL_CL_GUARDED(clFinish, (queue));
		//vec__zero(temp3,(2*N*N),vec_zero,queue);
		//check accuracy
		accfail = false;
		if(fail == false)
		{
			// local error			
			vec__add(u1,u,temp3,1,-1,2*N*N,vec_add,queue);
			vec__add(temp3,fu0,temp3,1,-kk,2*N*N,vec_add,queue);


#if 0 
	CALL_CL_GUARDED(clFinish, (queue));
	CALL_CL_GUARDED(clEnqueueReadBuffer, (
        	queue, temp3, /*blocking*/ CL_TRUE, /*offset*/ 0,
       		2*N*N* sizeof(float), uu,
        	0, NULL, NULL));
	CALL_CL_GUARDED(clEnqueueReadBuffer, (
        	queue, fu0, /*blocking*/ CL_TRUE, /*offset*/ 0,
       		2*N*N* sizeof(float), uuu,
        	0, NULL, NULL));
	printf("kk= %f\n",kk);

	/*for(int i =0; i<  N; i++)
	{
		printf("a%f+ i*",a[2*i]);		
		printf("%f\n",a[2*i+1]);
	}*/
	int T = 10<N? 10:N ;
	for(int i =0; i<  T; i++)
	{
		printf("%f + i*",uu[2*i]);		
		printf("%f\t",uu[2*i+1]);
		printf("%f + i*",uuu[2*i]);		
		printf("%f\n",uuu[2*i+1]);
	}

	#endif 
			
			Xi =  residual(temp3,temp9,resid,resid_init,queue,N*N);
			//Xi =0.1 * sigma;
			if (Xi > sigma)
			{
				printf("Accuracy lost, failing time step\n");
				printf("Xi = %f, sigma = %f\n",Xi,sigma);
				accfail = true;
			}
		}
		// check if time step failed
		if(fail == true || accfail == true)
		{
			if (fail == true || Xi/sigma >2)
				kk = kk * kfact2;
			else
				kk = kk * ksafety * sqrt(sigma/Xi);
		
			if(kk < mink)
			{
				epic_fail = true;
				printf("epic fail!!!!!!\n");
			}
		}
		else
		{

		
			clEnqueueCopyBuffer(queue,u1,u,
				0,
				0,
				sizeof(float)*N*N*2,0,NULL,NULL);

			clEnqueueCopyBuffer(queue,fu1,fu0,
				0,
				0,
				sizeof(float)*N*N*2,0,NULL,NULL);
			tt += kk;
			printf("time = %f!!!!!!!!!!!!!!!!!!!!!\n",tt);
			// estimate next time step
			kknew = kk * ksafety * sqrt(sigma/Xi);
			if( kknew > kk * kfact)
				kknew = kk * kfact;
			if((float)param.nloc/param.maxN > Nfact || (float)param.cgloc/param.maxCG>CGfact)
				kknew = kk < kknew? kk : kknew;
			kk = kknew;
			E0 =E1;
		}

				
	}



#if 1 
	CALL_CL_GUARDED(clFinish, (queue));
	CALL_CL_GUARDED(clEnqueueReadBuffer, (
        	queue, u, /*blocking*/ CL_TRUE, /*offset*/ 0,
       		2*N*N* sizeof(float), uu,
        	0, NULL, NULL));



	int T = 10<N? 10:N ;
	for(int i =0; i<  T; i++)
	{
		printf("%f + i*",uu[2*i]);		
		printf("%f\t",uu[2*i+1]);

	}

#endif 

	CALL_CL_GUARDED(clReleaseMemObject, (u));
	CALL_CL_GUARDED(clReleaseMemObject, (fu0));
	CALL_CL_GUARDED(clReleaseMemObject, (u1));
	CALL_CL_GUARDED(clReleaseMemObject, (fu1));
	CALL_CL_GUARDED(clReleaseMemObject, (temp));
	CALL_CL_GUARDED(clReleaseMemObject, (temp2));
	CALL_CL_GUARDED(clReleaseMemObject, (temp3));
	CALL_CL_GUARDED(clReleaseMemObject, (temp4));
	CALL_CL_GUARDED(clReleaseMemObject, (temp5));
	CALL_CL_GUARDED(clReleaseMemObject, (temp6));
	CALL_CL_GUARDED(clReleaseMemObject, (temp7));
	CALL_CL_GUARDED(clReleaseMemObject, (temp8));
	CALL_CL_GUARDED(clReleaseMemObject, (temp9));
	CALL_CL_GUARDED(clReleaseMemObject, (rhs));
	CALL_CL_GUARDED(clReleaseKernel, (fft_2D));
	CALL_CL_GUARDED(clReleaseKernel, (fft_2D_clean));
	CALL_CL_GUARDED(clReleaseKernel, (fft_init_w));
	CALL_CL_GUARDED(clReleaseKernel, (vec_add));
	CALL_CL_GUARDED(clReleaseKernel, (vec_zero));
	CALL_CL_GUARDED(clReleaseKernel, (resid));
	CALL_CL_GUARDED(clReleaseKernel, (resid_init));
	CALL_CL_GUARDED(clReleaseKernel, (reduct_mul));
	CALL_CL_GUARDED(clReleaseKernel, (reduct));
	CALL_CL_GUARDED(clReleaseKernel, (mat_trans));
	CALL_CL_GUARDED(clReleaseKernel, (mat_trans_3D));
	CALL_CL_GUARDED(clReleaseCommandQueue, (queue));
	CALL_CL_GUARDED(clReleaseContext, (ctx));

}



