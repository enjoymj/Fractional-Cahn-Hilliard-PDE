//#pragma OPENCL EXTENSION cl_amd_fp64 : enable

#ifndef M_PI
#define M_PI 3.14156265358979323846
#endif

inline FFT2(float2 * v)
{
	float2 v0 = v[0];
	v[0] = v0 + v[1];
	v[1] = v0 - v[1];
}
//Stockham radix-2 fft
__kernel void fft1D(
    __global float2 *a,
    __global float2 *b, 
    int N,
    int Ns,
    int direction,
    int offset_line)
{
	
	int gid = get_global_id(0);
	
	//radix-2 fft	
	float2 v[2];
	int idxS = gid;
	float xx;
	float yy;
	float angle = -2 *M_PI*(gid % Ns)/(Ns * 2) * direction;
	for(int  r = 0; r < 2; r++)
	{
		v[r] = a[offset_line * N +idxS + r * N / 2];
		xx = v[r].x;
		yy = v[r].y;
		v[r] =(float2) (xx* cos(r*angle)- yy* sin(r *angle), xx * sin(r * angle) + yy * cos(r * angle));
		
		
	}
	FFT2(v);
	int idxD = (gid / Ns)*Ns*2 + gid % Ns;

	barrier(CLK_GLOBAL_MEM_FENCE);
	for (int r =0; r< 2;r++)
	{
		b[offset_line * N + idxD + r * Ns] = v[r];
	}
	

	
}
