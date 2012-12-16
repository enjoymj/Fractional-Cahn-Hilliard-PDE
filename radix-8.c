
#include "timing.h"

#include "ppm.h"

#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#ifndef M_PI
#define M_PI 3.14156265358979323846
#endif



void FFT2(float * v)
{
	float v0 = v[0];
	float v00 = v[1];
	v[0] = v0 + v[2];
	
	v[1] = v00 + v[3];
	v[2] = v0 - v[2];
	v[3] = v00 - v[3];
	
}

void FFT4(float * v)
{
	float x = v[0];
	float y = v[1];


	
	v[0] = x + v[4];
	
	v[1] = y + v[5];
	v[4] = x - v[4];
	v[5] = y - v[5];
	
	x = v[2];
	y = v[3];
	v[2] = x + v[6];
	
	v[3] = y + v[7];
	v[6] = x - v[6];
	v[7] = y - v[7];


	x = v[0];
	y = v[1];

	v[0] = x + v[2];
	
	v[1] = y + v[3];
	v[2] = x - v[2];
	v[3] = y - v[3];

	x = v[4];
	y = v[5];
	
	v[4] = x + v[7];
	
	v[5] = y - v[6];
	v[6] = x - v[7];	
	v[7] = y + v[6];


	x = v[4];
	y= v[5];
	v[4] = v[2];
	v[5] = v[3];
	v[2] = x;
	v[3] = y;
	
}

void FFT8(float * v)
{
	float x = v[0];
	float y = v[1];

	float sr = sqrt(2)/2;
	
	v[0] = x + v[8];
	
	v[1] = y + v[9];
	v[8] = x - v[8];
	v[9] = y - v[9];
	
	x = v[4];
	y = v[5];
	v[4] = x + v[12];
	
	v[5] = y + v[13];
	v[12] = x - v[12];
	v[13] = y - v[13];


	x = v[0];
	y = v[1];

	v[0] = x + v[4];
	
	v[1] = y + v[5];
	v[4] = x - v[4];
	v[5] = y - v[5];

	x = v[8];
	y = v[9];
	
	v[8] = x + v[13];
	
	v[9] = y - v[12];
	v[12] = x - v[13];	
	v[13] = y + v[12];


	x = v[8];
	y= v[9];
	v[8] = v[4];
	v[9] = v[5];
	v[4] = x;
	v[5] = y;

	x = v[2];
	y = v[3];


	
	v[2] = x + v[10];
	
	v[3] = y + v[11];
	v[10] = x - v[10];
	v[11] = y - v[11];
	
	x = v[6];
	y = v[7];
	v[6] = x + v[14];
	
	v[7] = y + v[15];
	v[14] = x - v[14];
	v[15] = y - v[15];


	x = v[2];
	y = v[3];

	v[2] = x + v[6];
	
	v[3] = y + v[7];
	v[6] = x - v[6];
	v[7] = y - v[7];

	x = v[10];
	y = v[11];
	
	v[10] = x + v[15];
	
	v[11] = y - v[14];
	v[14] = x - v[15];	
	v[15] = y + v[14];


	x = v[10];
	y= v[11];
	v[10] = v[6];
	v[11] = v[7];
	v[6] = x;
	v[7] = y;

	float b[8];
	//float * d;
	b[0] = v[0];
	b[1] = v[1];
	
	
	v[0] += v[2];
	v[1] += v[3];

	b[2] = v[8];
	b[3] = v[9];

	v[8] = b[0]-v[2];
	v[9] = b[1]-v[3];

	b[0] = v[10];
	b[1] = v[11];

	v[2] = v[4] + sr * v[6]+ sr * v[7];
	v[3] = v[5] + sr * v[7] - sr * v[6];

	v[10] = v[4] -sr * v[6] -sr * v[7];
	v[11] = v[5] - sr * v[7]+sr * v[6];

	v[4] = b[2] + b[1];
	v[5] = b[3] - b[0];

	v[6] = v[12] -sr *v[14] + sr * v[15];
	v[7] = v[13] - sr  * v[14] -sr * v[15];
	
	b[4] = v[12];
	b[5] = v[13];

	b[6] = v[14];
	b[7] = v[15];

	v[12] = b[2] - b[1];
	v[13] = b[3] + b[0];

	v[14] = b[4] + sr *b[6] - sr * b[7];
	v[15] = b[5] + sr  * b[6] + sr * b[7];
	
	
}



void fftiteration(
	int j,
    int N,
    int Ns,
    float *a,
    float *b )
{
	
	int gid = j;
	
	//radix-8 fft	
	float v[16];
	int idxS = gid *2 ;
	float x;
	float y;
	int mask = Ns -1 ;
	float angle = -2 *M_PI*(gid & mask)/(Ns * 8) ;
	float s = sin(angle);
	float c = cos(angle);
	float s2 = 2 * s *c;
	float c2 = 2 * c * c -1;
	float s3 = 3 *s *c*c -s * s* s;
	float c3 = c*c*c -3*s*s*c;
	/*for(int r = 0; r< 8;r++)
	{
		x = v[2*r] = a[idxS + r * N / 4];
		y = v[2*r+1] = a[idxS + r * N / 4 + 1];
		v[2*r] = x * cos(r*angle) - y *sin(r*angle);
		v[2*r + 1 ] = x * sin(r*angle) + y* cos(r*angle);
	}*/
	v[0] = a[idxS];
	v[1] = a[idxS +1];


	x = v[2] = a[idxS +  N / 2];
	y = v[3] = a[idxS +  N / 2+1];

	v[2] = x* c- y* s; 
	v[3 ]= x * s + y * c;
	x = v[4] = a[idxS +  N ];
	y = v[5] = a[idxS +  N +1];

	v[4] = x* c2- y* s2; 
	v[5 ]= x * s2 + y * c2;
	x = v[6] = a[idxS + 3 * N / 2];
	y = v[7] = a[idxS + 3 * N / 2+1];

	v[6] = x* c3- y* s3; 
	v[7 ]= x * s3+ y * c3;
	
	x = v[8] = a[idxS + 4 * N / 4];
	y = v[9] = a[idxS + 4 * N / 4 + 1];
	v[8] = x * (2*c2*c2 -1) - y *2*c2 *s2;
	v[9] = x * 2*c2 *s2 + y* (2*c2*c2 -1);

	x = v[10] = a[idxS + 5 * N / 4];
	y = v[11] = a[idxS + 5 * N / 4 + 1];
	v[10] = x * (c3*c2-s3*s2) - y *(c3*s2 +c2*s3);
	v[11 ] = x * (c3*s2 +c2*s3) + y* (c3*c2-s3*s2);

	x = v[12] = a[idxS + 6 * N / 4];
	y = v[13] = a[idxS + 6 * N / 4 + 1];
	v[12] = x * (2 * c3 * c3 -1) - y *(2 * s3 *c3);
	v[13] = x * (2 * s3 *c3) + y*(2 * c3 * c3 -1);

	x = v[14] = a[idxS + 7 * N / 4];
	y = v[15] = a[idxS + 7 * N / 4 + 1];
	v[14] = x * (c3 *(2*c2*c2 -1)-s3 * 2*c2 *s2) - y *(s3*(2*c2*c2 -1) + c3*2*c2 *s2);
	v[15] = x * (s3*(2*c2*c2 -1) + c3*2*c2 *s2) + y* (c3 *(2*c2*c2 -1)-s3 * 2*c2 *s2);



	FFT8(v);

	int idxD = (gid / Ns)*Ns* 8 + (gid & mask);

	
	b[2*idxD ] = v[0];
	b[2*idxD +1] = v[1];
	b[2 *(idxD + Ns)] = v[2];
	b[2*(idxD + Ns)+1] = v[3];
	b[2*idxD + 4 * Ns] = v[4];
	b[2*idxD + 4 * Ns+1] = v[5];
	b[2*idxD + 6 * Ns] = v[6];
	b[2*idxD + 6 * Ns+1] = v[7];
	b[2*idxD + 8 * Ns] = v[8];
	b[2*idxD + 8 * Ns +1] = v[9];
	b[2*idxD + 10 * Ns] = v[10];
	b[2*idxD + 10 * Ns+1] = v[11];
	b[2*idxD + 12 * Ns] = v[12];
	b[2*idxD + 12 * Ns+1] = v[13];
	b[2*idxD + 14 * Ns] = v[14];
	b[2*idxD + 14 * Ns+1] = v[15];

	/*for(int r =0 ; r<8; r++)
																					
	{	
		b[2* idxD +r *2* Ns] =v[2*r];
		b[2* idxD +r *2* Ns+1] =v[2*r+1];
	}
*/
	
}


void main(int argc, char** argv)
{
	int  k  = atoi(argv[1]);
	int  N  = pow(2,k);


	float * a = (float *) malloc(sizeof(float)* N* 2);
	float * b = (float *) malloc(sizeof(float)* N * 2);
	float * c = (float *) malloc(sizeof(float)* N * 2);
	float p = 2*M_PI ;	
	for (int i =0; i< N; i++)
	{
		a[2*i] = 1;
		a[2*i+1] = 0;
		b[2*i] = 0;
		b[2*i+1] = 0;
	}
	int Ns;

	float *d;
	double elapsed;
	timestamp_type time1, time2;

	get_timestamp(&time1);



	for (int j = 0; j<N/8; j++)
	{
		
		int gid = j;

		//radix-8 fft	
		float v[16];
		int idxS = gid *2 ;
		float x;
		float y;
		
		//float angle = -2 *M_PI*(gid & mask)/(Ns * 8) ;
		//float s = sin(angle);
		//float c = cos(angle);
		/*for(int r = 0; r< 8;r++)
		{
			v[2*r] = a[idxS + r * N / 4];
			v[2*r+1] = a[idxS + r * N / 4 + 1];

		}*/

		v[0] = a[idxS ];
		v[1] = a[idxS + 1];

		v[2] = a[idxS +  N / 4];
		v[3] = a[idxS + N / 4 + 1];

		v[4] = a[idxS + N / 2];
		v[5] = a[idxS + N / 2 + 1];

		v[6] = a[idxS + 3 * N / 4];
		v[7] = a[idxS + 3 * N / 4 + 1];

		v[8] = a[idxS +  N ];
		v[9] = a[idxS + N  + 1];

		v[10] = a[idxS + 5 * N / 4];
		v[11] = a[idxS + 5 * N / 4 + 1];

		v[12] = a[idxS + 3 * N / 2];
		v[13] = a[idxS + 3 * N / 2 + 1];

		v[14] = a[idxS + 7 * N / 4];
		v[15] = a[idxS + 7 * N / 4 + 1];

		FFT8(v);

		int idxD = gid * 8 ;

	


		b[2*idxD ] = v[0];
		b[2*idxD +1] = v[1];
		b[2 *(idxD + Ns)] = v[2];
		b[2*(idxD + Ns)+1] = v[3];
		b[2*idxD + 4 * Ns] = v[4];
		b[2*idxD + 4 * Ns+1] = v[5];
		b[2*idxD + 6 * Ns] = v[6];
		b[2*idxD + 6 * Ns+1] = v[7];
		b[2*idxD + 8 * Ns] = v[8];
		b[2*idxD + 8 * Ns +1] = v[9];
		b[2*idxD + 10 * Ns] = v[10];
		b[2*idxD + 10 * Ns+1] = v[11];
		b[2*idxD + 12 * Ns] = v[12];
		b[2*idxD + 12 * Ns+1] = v[13];
		b[2*idxD + 14 * Ns] = v[14];
		b[2*idxD + 14 * Ns+1] = v[15];


	}
	
	d = a;
	a = b;
	b = d;
	
	

	for( Ns = 8;Ns < N; Ns *= 8 )
	{
		for (int j = 0; j<N/8; j++)
		{
			fftiteration(j,N,Ns,a,b);

		}
		
		d = a ;
		a = b;
		b = d;
		//printf("ok\n");

	}

	get_timestamp(&time2);
	elapsed = timestamp_diff_in_seconds(time1,time2);
	printf("1D %f s\n", elapsed);
	printf("archieve %f GFLOP/s\n", 4*N*k/elapsed/1e9);
	int T = 10<N? 10:N ;
	for(int i =0; i<  T; i++)
	{
		printf("%f + i*",a[2*i]);		
		printf("%f\n",a[2*i+1]);
		//printf("%f + i*",c[2*i]/N/N);		
		//printf("%f\n",c[2*i+1]/N/N);
	}


}


