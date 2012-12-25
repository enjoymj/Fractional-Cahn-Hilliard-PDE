//#pragma OPENCL EXTENSION cl_amd_fp64 : enable

#ifndef M_PI
#define M_PI 3.14156265358979323846
#endif


#define BLK 16

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
__kernel void fft1D(
    __global float2 *a,
    __global float2 *b, 
    int N,
    int Ns,
    int direction,
    int offset_line)
{
	__local float2 l_a[64];
	int gid = get_global_id(0);
	int t = get_local_id(0);
	//radix-4 fft	
	float2 v[4];
	int idxS = gid;
	float2 x;
	//float2 y;
	int mask = Ns -1;
	float angle = -2 *(gid & mask)/(Ns<<2) * direction;
	float s = sinpi(angle);
	float c = cospi(angle);
	int offset = offset_line * N;

	v[0] = a[offset +idxS];
	v[1] = a[offset+idxS+ N/4];

	x = v[1];
	
	
	v[1] = (float2)(x.x* c- x.y* s, x.x * s + x.y * c);
	
	v[2] = a[offset+ idxS +  N/2 ];
	
	x = v[2];
	
	//y = (float2)(2*c*c -1, 2*c*s);

	float cc = 2*c*c -1;
	float ss = 2*c*s;
	v[2] = (float2) (x.x* cc- x.y* ss, x.x * ss + x.y * cc); 
	//v[2] = (float2) (x.x* (2*c*c -1)- x.y* 2*c*s, x.x * 2*c*s + x.y * (2*c*c -1)); 
	
	v[3] = a[offset +idxS + 3 * N / 4];
	x = v[3];
	float ccc = cc*c -ss*s;
	float sss = cc*s +ss*c;
	v[3] = (float2) (x.x* ccc- x.y* sss, x.x * sss + x.y * ccc);
	//y = (float2)(c*c*c -3*s*s*c, 3 *s *c*c -s * s* s);
	//v[3] = (float2) (x.x* (c*c*c -3*s*s*c)- x.y* (3 *s *c*c -s * s* s), x.x * (3 *s *c*c -s * s* s) + x.y * (c*c*c -3*s*s*c));
	
	//v[3] = (float2) (x.x* y.x- x.y* y.y, x.x * y.y + x.y * y.x);

	FFT4(v,direction);



	//int idxD = (gid / Ns)*Ns*4 + (gid & mask);
	int idxD = (t/Ns)*Ns*4 +(t & mask);
	//exchange(v,idxD,Ns,t,BLK);
	//barrier(CLK_LOCAL_MEM_FENCE);

		//int i = idxD +r*Ns;
		l_a[idxD] = v[0];
		l_a[idxD +Ns] = v[1];
		l_a[idxD +2*Ns] = v[2];
		l_a[idxD +3*Ns] = v[3];
	
	barrier(CLK_LOCAL_MEM_FENCE);
	//int i = t*4 + r;
		v[0] = l_a[t*4 ];
		v[1] = l_a[t*4 + 1];
		v[2] = l_a[t*4 + 2];
		v[3] = l_a[t*4 + 3];
	
	//barrier(CLK_GLOBAL_MEM_FENCE);

	idxD = get_group_id(0) *4*BLK + t*4;
	b[offset + idxD ] = v[0];
	b[offset+ idxD + 1] = v[1];
	b[offset+ idxD + 2 ] = v[2];
	b[offset + idxD + 3 ] = v[3];
	
	

	
}
