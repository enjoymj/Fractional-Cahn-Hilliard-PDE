//#pragma OPENCL EXTENSION cl_amd_fp64 : enable

#ifndef M_PI
#define M_PI 3.14156265358979323846
#endif

inline void FFT2(float2 * v)
{
	float2 v0 = v[0];
	v[0] = v0 + v[1];
	v[1] = v0 - v[1];
}

inline void FFT4(float2 * v,int direction)
{
	float2 x = v[0];
	


	
	v[0] = x + v[2];
	
	
	v[2] = x - v[2];
	
	
	x = v[1];
	
	v[1] = x + v[3];
	
	
	v[3] = x - v[3];
	


	x = v[0];

	v[0] = x + v[1];
	
	v[1] = x - v[1];
	

	x = v[2];
	
	
	v[2] = (float2)(x.x  + v[3].y, x.y-v[3].x);
	v[3] = (float2)(x.x - v[3].y, x.y+v[3].x);

	x = v[2];
	v[2] = v[1];
	v[1] = x;

	if(direction == -1)
	{
		x = v[3];
		v[3] = v[1];
		v[1] = x;
	}
	
}
//Stockham radix-4 fft
__kernel void fft1D_init(
    __global float2 *a,
    __global float2 *b, 
    int N,
    int Ns,
    int direction,
    int offset_line,
    int option)
{
	

	int gid = get_global_id(0);
	
	//radix-4 fft	
	float2 v[4];
	int idxS = gid;
	//float2 x;
	
	//int mask = Ns -1;
	//float angle = -2 *M_PI*(gid & mask)/(Ns * 4) ;
	//float s = 0;
	//float c = 1;
	int offset =offset_line *N;
	v[0] = a[offset +idxS];
	//printf("%f!!!!!!!!!!!!\n",v[0].x);
	v[1] = a[offset +idxS+ N/4];


	
	v[2] = a[offset+ idxS +  N/2 ];
	

	
	v[3] = a[offset +idxS + 3 * N / 4];
	if(0 == option)
	{1+1;}
	else if(1 == option)
	{
		v[0].x = pow(v[0].x,3)- v[0].x;
		//printf("%f!\n",v[0].x);
		v[1].x = pow(v[1].x,3)- v[1].x;
		v[2].x = pow(v[2].x,3)- v[2].x;
		v[3].x = pow(v[3].x,3)- v[3].x;
	}

	FFT4(v,direction);
	int idxD = gid*4 ;

	//barrier(CLK_GLOBAL_MEM_FENCE);

	
	b[offset + idxD ] = v[0];
	b[offset+ idxD +  Ns] = v[1];
	b[offset+ idxD + 2 * Ns] = v[2];
	b[offset+ idxD + 3 * Ns] = v[3];
	
	

	
}
