#include<stdbool.h>
#include"cl-helper.h"
#ifndef PI
#define PI 3.14156265358979323846
#endif


struct parameter
{
	float h;
        int N;
	float epsilon;
	float s;	
        //coutourv = linspace(-1,1,11);

	// maximum total number of CG or fixed point interations per time step
	int maxCG;

	// maximum number of Newton steps per time step
	int maxN;

	float Ntol;
	float cgtol;
	//param.xx = xx;
	//param.yy = yy;
	int cgloc;
	int nloc;

};

void vec__add(cl_mem a, cl_mem b, cl_mem c, float a_mult, float b_mult, 
		long n, cl_kernel knl, cl_command_queue queue);

void mat_etr_mul(cl_mem a, cl_mem b, cl_mem c, 
		long n, cl_kernel knl, cl_command_queue queue);

void fft_1D(cl_mem a,cl_mem b,cl_mem c, int N, cl_kernel init, cl_kernel knl,cl_command_queue queue,int direction,int offset_line);

void mat__trans(cl_mem a, cl_mem b, int N, cl_kernel mat_trans, cl_command_queue queue,int option, float epsilon,float k,float s);

void fft2D(cl_mem a, cl_mem c, cl_mem b,cl_mem d, int N, cl_kernel init_big,
		cl_kernel clean,cl_kernel mat_trans, cl_kernel mat_trans_3D,
		cl_command_queue queue,int direction);

//fft first arg devided by q entry-wise and save to second arg
void fft_d_q(cl_mem a,cl_mem c,cl_mem b,cl_mem d, int N,float epsilon,float k,float s, cl_kernel init_big,
		cl_kernel clean,cl_kernel mat_trans,cl_kernel mat_trans_3D,cl_command_queue queue);

//real(ifft2(fft2(pk)./nlap_s2));
void fft_d_nlaps2(cl_mem a,cl_mem c,cl_mem b,cl_mem d,int N,float epsilon,float k,float s, cl_kernel init_big,
		cl_kernel clean,cl_kernel mat_trans,cl_kernel mat_trans_3D,cl_command_queue queue);

//real(ifft2(fft2(pk).*sharmonic));
void fft_shar(cl_mem a ,cl_mem c ,cl_mem b,cl_mem d,int N,float epsilon,float k,float s, cl_kernel init_big,
		cl_kernel clean,cl_kernel mat_trans,cl_kernel mat_trans_3D,cl_command_queue queue);

// fft(u.^3-u) .*nlap_s
void fft_w_orig(cl_mem a, cl_mem c, cl_mem b,cl_mem d, int N,float epsilon,float k,float s, cl_kernel init_big,
		cl_kernel clean,cl_kernel mat_trans,cl_kernel mat_trans_3D, cl_command_queue queue);


//fft((3*u.^2 -1) .* pk).*nlap_s
void fft_w(cl_mem a, cl_mem b, cl_mem c,/*result */cl_mem d,int N,float epsilon,float k,float s, 
		cl_kernel init_big,cl_kernel init_w,cl_kernel clean,
		cl_kernel mat_trans,cl_kernel mat_trans_3D,cl_command_queue queue);




void vec__zero(cl_mem a, int n, cl_kernel vec_zero, cl_command_queue queue);


void chcg(float k,struct parameter * p_param,  cl_mem temp,cl_mem rhs, cl_mem temp2, bool *fail,
		cl_mem temp3,cl_mem temp4,cl_mem temp5,cl_mem temp6,cl_mem temp7,cl_mem temp8,cl_mem temp9,cl_kernel fft_2D,
		cl_kernel fft_2D_clean, cl_kernel fft_init_w,cl_kernel vec_add, cl_kernel vec_zero,
		cl_kernel mat_trans,cl_kernel mat_trans_3D,cl_kernel reduct,cl_kernel reduct_init,
		cl_kernel reduct_mul,cl_kernel resid, cl_kernel resid_init,cl_command_queue queue);

void chstep(cl_mem u, cl_mem fu0, cl_mem u1, cl_mem rhs, cl_mem fu1, cl_mem temp,  cl_mem temp2,cl_mem temp3,
		cl_mem temp4,cl_mem temp5,cl_mem temp6,cl_mem temp7,cl_mem temp8,cl_mem temp9,
		int N, bool * fail, float k, struct parameter* p_param, 
		cl_kernel fft_2D,cl_kernel fft_2D_clean, cl_kernel fft_inti_w,cl_kernel vec_add, 
		cl_kernel vec_zero,cl_kernel mat_trans,cl_kernel mat_trans_3D,cl_kernel reduct,cl_kernel reduct_mul, 
		cl_kernel reduct_init,  cl_kernel resid, cl_kernel resid_init,cl_command_queue queue);


void frhs(/*variable*/cl_mem temp,  /*result*/ cl_mem temp2, 
		 cl_mem temp3, cl_mem temp4,cl_mem temp9, 
		struct parameter* p_param, cl_kernel init_big,cl_kernel clean,
		cl_kernel mat_trans,cl_kernel mat_trans_3D,
		 cl_kernel vec_add, cl_command_queue queue);



float reduction_mult(cl_mem a,cl_mem b, cl_mem c, int N, cl_kernel reduct_mul,cl_kernel reduct, cl_command_queue queue);

float reduction(cl_mem a,cl_mem b, int N,cl_kernel reduct_init,cl_kernel reduct, cl_command_queue queue);

float residual(cl_mem a, cl_mem b, cl_kernel resid, cl_kernel resid_init, cl_command_queue queue, int N);


void fft_d_y(cl_mem a,cl_mem c,cl_mem b,cl_mem d,int N,float epsilon,float k,float s, cl_kernel init_big,
		cl_kernel clean,cl_kernel mat_trans,cl_kernel mat_trans_3D,cl_command_queue queue);
void fft_d_x(cl_mem a,cl_mem c,cl_mem b,cl_mem d,int N,float epsilon,float k,float s, cl_kernel init_big,
		cl_kernel clean,cl_kernel mat_trans,cl_kernel mat_trans_3D,cl_command_queue queue);


float energy(cl_mem a, cl_mem b, cl_mem c,cl_mem d, cl_mem e,cl_mem f,float k, 
		struct parameter* p_param, cl_kernel init_big,cl_kernel clean,
		cl_kernel mat_trans,cl_kernel mat_trans_3D, cl_kernel reduct_eng, 
		cl_kernel reduct,cl_command_queue queue);

float reduct_energy(cl_mem a,cl_mem b, cl_mem c,cl_mem d, int N, float epsilon,
			 cl_kernel reduct_eng,cl_kernel reduct, cl_command_queue queue);


void fft2D_transpose(cl_mem a, cl_mem c, cl_mem b,cl_mem d, int N, cl_kernel init_big,
		cl_kernel clean,cl_kernel mat_trans, cl_kernel mat_trans_3D, cl_command_queue queue,int direction,int y);






