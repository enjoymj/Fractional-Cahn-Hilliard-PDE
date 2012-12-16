
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

//Stockham radix-4 fft
void fftiteration(
	int j,
    int N,
    int Ns,
    float *a,
    float *b )
{
	
	int gid = j;
	
	//radix-4 fft	
	float v[8];
	int idxS = gid *2 ;
	float x;
	float y;
	int mask = Ns -1 ;
	float angle = -2 *M_PI*(gid & mask)/(Ns * 4) ;
	float s = sin(angle);
	float c = cos(angle);

	v[0] = a[idxS];
	v[1] = a[idxS +1];


	x = v[2] = a[idxS +  N / 2];
	y = v[3] = a[idxS +  N / 2+1];

	v[2] = x* c- y* s; 
	v[3 ]= x * s + y * c;
	x = v[4] = a[idxS +  N ];
	y = v[5] = a[idxS +  N +1];

	v[4] = x* (2*c*c-1)- y* 2*c*s; 
	v[5 ]= x * 2*c*s + y * (2*c*c-1);
	x = v[6] = a[idxS + 3 * N / 2];
	y = v[7] = a[idxS + 3 * N / 2+1];

	v[6] = x* (c*c*c -3*s*s*c)- y* (3 *s *c*c -s * s* s); 
	v[7 ]= x * (3 *s *c*c -s * s* s) + y * (c*c*c -3*s*s*c);
	

	x = v[0];
	y = v[1];


	
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

	int idxD = (gid / Ns)*Ns* 4 + (gid & mask);

	
	b[2*idxD ] = v[0];
	b[2*idxD +1] = v[1];
	b[2 *(idxD + Ns)] = v[2];
	b[2*(idxD + Ns)+1] = v[3];
	b[2*idxD + 4 * Ns] = v[4];
	b[2*idxD + 4 * Ns+1] = v[5];
	b[2*idxD + 6 * Ns] = v[6];
	b[2*idxD + 6 * Ns+1] = v[7];



	
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


	for (int j = 0; j<N/4; j++)
	{
		
		 
		int gid = j;
	
		//radix-4 fft	
		float v[8];
		int idxS = gid *2 ;
		float xx;
		float yy;
		//int mask = Ns -1 ;
		//float angle = -2 *M_PI*(gid & mask)/(Ns * 4) ;
		//float s = 0;
		//float c = 1;

		v[0] = a[idxS];
		v[1] = a[idxS +1];


		v[2] = a[idxS +  N / 2];
		v[3] = a[idxS +  N / 2+1];


		v[4] = a[idxS +  N ];
		v[5] = a[idxS +  N +1];

		v[6] = a[idxS + 3 * N / 2];
		v[7] = a[idxS + 3 * N / 2+1];

	
		
		FFT4(v);
		int idxD = gid *4 ;

	
		b[2*idxD ] = v[0];
		b[2*idxD +1] = v[1];
		b[2*idxD + 2] = v[2];
		b[2*idxD + 3] = v[3];
		b[2*idxD + 4] = v[4];
		b[2*idxD + 5] = v[5];
		b[2*idxD + 6 ] = v[6];
		b[2*idxD + 7] = v[7];
	}
	
	d = a;
	a = b;
	b = d;

	/*for (int j = 0; j<N/2; j++)
	{
		
		int mask = 1; 
	
		//radix-2 fft	
		//float v[4];
		
		//float xx;
		//float yy;
		
		float angle = -2 *M_PI*(j & mask)/4 ;


		v[0] = a[2 * j ];
		v[1] = a[2 * j +1];
	
		v[2] = a[2 * j + N ];
		v[3] = a[2 * j + N +1];

		if(j & mask == 1)
		{
						
			xx = v[2];
			yy = v[3];
		
			v[2] = - yy; 
			v[3]=  xx ;
		}

		xx = v[0];
		yy = v[1];

		v[0] = xx + v[2];
		v[1] = yy + v[3];
		v[2] = xx - v[2];
		v[3] = yy - v[3];

		int idxD = (j >> 1) << 2 + (j & mask);

		int k =  idxD << 1 ;
	

		b[k] = v[0];
		b[k+1] = v[1];
		b[k+4] = v[2];
		b[k+5] = v[3];

	

	}
	
	d = a;
	a = b;
	b = d;
	
	*/

	for( Ns = 4;Ns < N; Ns <<= 2 )
	{
		for (int j = 0; j<N/4; j++)
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


