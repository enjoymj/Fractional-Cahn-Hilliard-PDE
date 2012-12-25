#include<math.h>
#include<stdio.h>
#include<stdbool.h>
#include"cl-helper.h"

#include"ch.h"



void chstep(cl_mem u, cl_mem fu0, cl_mem u1, cl_mem rhs, cl_mem fu1, cl_mem temp,  cl_mem temp2,cl_mem temp3,
		cl_mem temp4,cl_mem temp5,cl_mem temp6,cl_mem temp7,cl_mem temp8,cl_mem temp9,
		int N, bool * fail, float k, struct parameter* p_param, 
		cl_kernel fft_2D,cl_kernel fft_2D_clean, cl_kernel fft_inti_w,cl_kernel vec_add, 
		cl_kernel vec_zero,cl_kernel mat_trans,cl_kernel mat_trans_3D,cl_kernel reduct,cl_kernel reduct_mul, 
		cl_kernel reduct_init,  cl_kernel resid, cl_kernel resid_init,cl_command_queue queue)
{
	
	//following equal to rhs = fnew - frhs(ustar+k*alpha*fnew,param,symbol);
	vec__add(u, fu0, temp, 1, k , 
		2*N*N, vec_add, queue);
	*fail = false;
	#if 0
	float test;
	CALL_CL_GUARDED(clFinish, (queue));
	CALL_CL_GUARDED(clEnqueueReadBuffer, (
        	queue, temp, /*blocking*/ CL_TRUE, /*offset*/ 0,
       		sizeof(float), &test,
        	0, NULL, NULL));
	

		printf("test success and r0 is %f \n",test);		
	

	#endif
	frhs(temp,temp2,temp3,temp4,temp9,p_param,fft_2D,fft_2D_clean,mat_trans,mat_trans_3D,
		 vec_add, queue);

        vec__add(fu0,temp2,rhs,1,-1,2*N*N,vec_add,queue);


	#if 0
	float test;
	CALL_CL_GUARDED(clFinish, (queue));
	CALL_CL_GUARDED(clEnqueueReadBuffer, (
        	queue, rhs, /*blocking*/ CL_TRUE, /*offset*/ 0,
       		sizeof(float), &test,
        	0, NULL, NULL));
	

		printf("test success and rhs = %f \n",test);		
	

	#endif
	
	clEnqueueCopyBuffer(queue,fu0,fu1,
		0,0,
		sizeof(float)*N*N*2,0,NULL,NULL);
	//vec_copy(fu0,fu1,2*N*N,p_knl, p_queue);
	

	float reside = 1;
	p_param->cgloc = 0;
        p_param->nloc = 0;
	//CALL_CL_GUARDED(clFinish, (queue));
	//printf("enter cg loop!\n");
        while( reside > p_param->Ntol && p_param->nloc < p_param->maxN && *fail == false)
	{
		chcg(k,p_param, temp, rhs,/*result*/  temp2, fail,
			temp3,temp4,temp5,temp6,temp7,temp8,temp9,fft_2D,
			fft_2D_clean,fft_inti_w,vec_add,vec_zero,mat_trans,mat_trans_3D,
			reduct,reduct_init,reduct_mul,resid,resid_init,queue);
		



		p_param->nloc ++ ;
		
		// following equal to fnew = fnew - delta;
		vec__add(fu1,temp2,fu1,1,-1,2*N*N,vec_add,queue);
		//printf("in cg loop!\n");

		if(k*residual(fu1,temp9,resid,resid_init,queue,N*N)>5)
		{
			*fail = true;
			printf("wild Newton step!!\n");
		}
		else
		{
			//following equal to rhs = fnew - frhs(ustar+k*alpha*fnew,param,symbol);
		
			//should check for wild Newton step
			vec__add(u, fu1, temp, 1, k , 2*N*N, vec_add, queue);




		frhs(temp,temp2,temp3,temp4,temp9,p_param,fft_2D,fft_2D_clean,mat_trans,mat_trans_3D,
		 vec_add, queue);

		vec__add(fu1,temp2,rhs,1,-1,2*N*N,vec_add,queue);


			#if 0
		float test;
		CALL_CL_GUARDED(clFinish, (queue));
		CALL_CL_GUARDED(clEnqueueReadBuffer, (
		queue, rhs, /*blocking*/ CL_TRUE, /*offset*/ 0,
		sizeof(float), &test,
		0, NULL, NULL));


		printf("in CHstep loop and temp2 = %f \n",test);		


		#endif
		reside = residual(rhs,temp9,resid,resid_init,queue,N*N);
//CALL_CL_GUARDED(clFinish, (queue));
//printf("Chstep resid  = %f \n",reside);
		}
	}
	if (reside > p_param->Ntol)
	{
		printf("maximum iterations exceeded\n");
		*fail = true;
 
	}
	if (p_param->cgloc > p_param->maxCG)
	{
		printf("maximum CG iteration exceeded\n");
		*fail = true;
	}
	//unew = u0 + k*fnew;
	vec__add(u, fu1, u1, 1, k , 2*N*N, vec_add, queue);

		#if 0
		float test;
		CALL_CL_GUARDED(clFinish, (queue));
		CALL_CL_GUARDED(clEnqueueReadBuffer, (
		queue, u, /*blocking*/ CL_TRUE, /*offset*/ 0,
		sizeof(float), &test,
		0, NULL, NULL));


		printf("in CHstep loop and unew = %f \n",test);		


		#endif

}
