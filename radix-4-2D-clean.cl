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
__kernel void fft2D_clean(
    __global const float2 *a,
    __global float2 *b, 
    int N,
    int Ns,
    int direction,
    int offset_line,
    int option)
{
	
	int gid = get_global_id(0);
	int l_gid = gid%(N/4);
	offset_line = gid/(N/4);	
	float2 v[4];
	int idxS = l_gid*4;	
	int offset =offset_line *N;

	if(N/64 ==4)
	{
		v[0] = a[offset +idxS];
		v[1] = a[offset +idxS+ 1];
		v[2] = a[offset+ idxS + 2 ];
		v[3] = a[offset +idxS + 3 ];
		FFT4(v,direction);
		b[offset + idxS ] = v[0];
		b[offset+ idxS+1] = v[1];
		b[offset+ idxS + 2] = v[2];
		b[offset+ idxS + 3] = v[3];

	}
	else if(N/64 ==16)
	{
		__local float2 l_a[16];		
		int lid =get_local_id(0);
		int grp = get_group_id(0);
		int l_grp = grp%(N/16);		
		v[0] = a[offset +l_grp*16 + lid];
		v[1] = a[offset +l_grp*16 + lid + 4];
		v[2] = a[offset +l_grp*16 + lid + 8];
		v[3] = a[offset +l_grp*16 + lid + 12];
		FFT4(v,direction);
		l_a[lid*4 ] = v[0];
		l_a[lid*4 + 1] = v[1];
		l_a[lid*4 + 2] = v[2];
		l_a[lid*4 + 3] = v[3];

		barrier(CLK_LOCAL_MEM_FENCE);

		int mask = 3;
		float angle = -(2.0 *(lid & mask))/16 * direction;
		float s = sinpi(angle);
		float c = cospi(angle);
		v[0] = l_a[lid];
		v[1] = l_a[lid+4];
		float2 x = v[1];
	
	
		v[1] = (float2)(x.x* c- x.y* s, x.x * s + x.y * c);
		v[2] = l_a[lid + 8];
		x = v[2];
	
	
		v[2] = (float2) (x.x* (2*c*c -1)- x.y* 2*c*s, x.x * 2*c*s + x.y * (2*c*c -1)); 
		
		v[3] = l_a[lid + 12];
		x = v[3];
		v[3] = (float2) (x.x* (c*c*c -3*s*s*c)- x.y* (3 *s *c*c -s * s* s),
			 x.x * (3 *s *c*c -s * s* s) + x.y * (c*c*c -3*s*s*c));
	
	

		FFT4(v,direction);

		
		l_a[lid] = v[0];
		l_a[lid +4] = v[1];
		l_a[lid+8] = v[2];
		l_a[lid +12] = v[3];
	
		barrier(CLK_LOCAL_MEM_FENCE);
	//int i = t*4 + r;
		v[0] = l_a[lid*4 ];
		v[1] = l_a[lid*4 + 1];
		v[2] = l_a[lid*4 + 2];
		v[3] = l_a[lid*4 + 3];

		b[offset+ l_grp*16 + lid*4] = v[0];
		b[offset+ l_grp*16 + lid*4+1] = v[1];
		b[offset+ l_grp*16 + lid*4+2] = v[2];
		b[offset+ l_grp*16 + lid*4+3] = v[3];
	}
	
}
