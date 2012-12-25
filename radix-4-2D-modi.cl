//#pragma OPENCL EXTENSION cl_amd_fp64 : enable

#define TWO_PI 6.28318530718

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
//Stockham radix-4 fft for size of 64
__kernel void fft2D_big(
    __global const float2 *a,
    __global const float2 *b, 
    __global float2 *m_c,
    int N,
    int Ns,
    int direction,
    int offset_line,
    int option)
{
	
	__local float2 l_a[64];
	__local float2 l_p[64];
	int gid = get_global_id(0);
	int t = get_local_id(0);
	int grp = get_group_id(0);	
	float2 v[4];
	int idxS = t;
	int l_grp = grp%(N/64);
	offset_line = grp/(N/64);
	int offset = offset_line * N;

	//save to size of 64 local array;
	l_a[t*4]=a[offset+t*4*N/64+l_grp];
	l_a[t*4+1]=a[offset+t*4*N/64+N/64+l_grp];
	l_a[t*4+2]=a[offset+t*4*N/64+2*N/64+l_grp];
	l_a[t*4+3]=a[offset+t*4*N/64+3*N/64+l_grp];
	l_p[t*4]=b[offset+t*4*N/64+l_grp];
	l_p[t*4+1]=b[offset+t*4*N/64+N/64+l_grp];
	l_p[t*4+2]=b[offset+t*4*N/64+2*N/64+l_grp];
	l_p[t*4+3]=b[offset+t*4*N/64+3*N/64+l_grp];
	barrier(CLK_LOCAL_MEM_FENCE);



	v[0] = l_a[idxS];
	v[1] = l_a[idxS+ 16];
	v[2] = l_a[idxS + 32];
	v[3] = l_a[idxS + 48];
		v[0].x = 3*pow(v[0].x,2)- l_p[idxS].x;
		//printf("%f!\n",v[0].x);
		v[1].x = 3*pow(v[1].x,2)- l_p[idxS+16].x;
		v[2].x = 3*pow(v[2].x,2)- l_p[idxS+32].x;
		v[3].x = 3*pow(v[3].x,2)- l_p[idxS+48].x;

	FFT4(v,direction);
	int idxD = t*4 ;

	//barrier(CLK_GLOBAL_MEM_FENCE);
	l_a[idxD] = v[0];
	l_a[idxD+1] = v[1];
	l_a[idxD+2] = v[2];
	l_a[idxD+3] = v[3];

	Ns =4;
	barrier(CLK_LOCAL_MEM_FENCE);
	
	int mask = 3;
	float angle = -2 *(gid & 3)/16 * direction;
	float s = sinpi(angle);
	float c = cospi(angle);
	idxS = t;

	v[0] = l_a[idxS];
	v[1] = l_a[idxS+ 16];

	float2 x = v[1];
	
	
	v[1] = (float2)(x.x* c- x.y* s, x.x * s + x.y * c);
	
	v[2] = l_a[idxS +  32 ];
	
	x = v[2];
	float cc = 2*c*c-1;
	float ss =2*c*s;
	//y = (float2)(2*c*c -1, 2*c*s);
	v[2] = (float2) (x.x* cc- x.y*ss, x.x * ss + x.y * cc); 
	
	v[3] = l_a[idxS +48];
	x = v[3];
	
	float ccc = cc*c-ss*s;
	float sss =cc*s +ss *c;
	//y = (float2)(c*c*c -3*s*s*c, 3 *s *c*c -s * s* s);
	v[3] = (float2) (x.x* ccc- x.y* sss, x.x * sss + x.y *ccc);
	
	//v[3] = (float2) (x.x* y.x- x.y* y.y, x.x * y.y + x.y * y.x);

	FFT4(v,direction);



	idxD = (t/Ns)*Ns*4 +(t & mask);

	barrier(CLK_LOCAL_MEM_FENCE);

		//int i = idxD +r*Ns;
		l_a[idxD] = v[0];
		l_a[idxD +Ns] = v[1];
		l_a[idxD +2*Ns] = v[2];
		l_a[idxD +3*Ns] = v[3];
	
	barrier(CLK_LOCAL_MEM_FENCE);

		Ns =16;
	//barrier(CLK_LOCAL_MEM_FENCE);
	
	 mask = 15;
	angle = -2 *(gid & 15)/64 * direction;
	s = sinpi(angle);
	c = cospi(angle);
	idxS = t;

	v[0] = l_a[idxS];
	v[1] = l_a[idxS+ 16];

	x = v[1];
	
	
	v[1] = (float2)(x.x* c- x.y* s, x.x * s + x.y * c);
	
	v[2] = l_a[idxS +  32 ];
	
	x = v[2];
	cc = 2*c*c-1;
	ss =2*c*s;
	//y = (float2)(2*c*c -1, 2*c*s);
	v[2] = (float2) (x.x* cc- x.y*ss, x.x * ss + x.y * cc); 
	//y = (float2)(2*c*c -1, 2*c*s);
	//v[2] = (float2) (x.x* (2*c*c -1)- x.y* 2*c*s, x.x * 2*c*s + x.y * (2*c*c -1)); 
	
	v[3] = l_a[idxS +48];
	x = v[3];
	ccc = cc*c-ss*s;
	sss =cc*s +ss *c;
	
	v[3] = (float2) (x.x* ccc- x.y* sss, x.x * sss + x.y *ccc);

	//y = (float2)(c*c*c -3*s*s*c, 3 *s *c*c -s * s* s);
	//v[3] = (float2) (x.x* (c*c*c -3*s*s*c)- x.y* (3 *s *c*c -s * s* s), x.x * (3 *s *c*c -s * s* s) + x.y * (c*c*c -3*s*s*c));
	
	//v[3] = (float2) (x.x* y.x- x.y* y.y, x.x * y.y + x.y * y.x);

	FFT4(v,direction);



	idxD = (t/Ns)*Ns*4 +(t & mask);

	barrier(CLK_LOCAL_MEM_FENCE);

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
//twiddle factor
	float tw =-2 *t*4*l_grp/N*direction;
	float inc = -(2.0*l_grp)/N*direction;
	s = sinpi(tw);
	c=  cospi(tw);
	v[0]=(float2)(v[0].x*c-v[0].y*s,v[0].x*s+v[0].y*c);
	s = sinpi(tw+inc);
	c=  cospi(tw+inc);
	v[1]=(float2)(v[1].x*c-v[1].y*s,v[1].x*s+v[1].y*c);
	s = sinpi(tw+2*inc);
	c=  cospi(tw+2*inc);
	v[2]=(float2)(v[2].x*c-v[2].y*s,v[2].x*s+v[2].y*c);
	s = sinpi(tw+3*inc);
	c=  cospi(tw+3*inc);
	v[3]=(float2)(v[3].x*c-v[3].y*s,v[3].x*s+v[3].y*c);





	idxD = l_grp + t*4*N/64;
	m_c[offset + idxD ] = v[0];
	m_c[offset+ idxD + N/64] = v[1];
	m_c[offset+ idxD + 2*N/64 ] = v[2];
	m_c[offset + idxD + 3*N/64 ] = v[3];
	
}
