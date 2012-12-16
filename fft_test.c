
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
//Stockham radix-2 fft
void fftiteration(
	int j,
    int N,
    int Ns,
    float *a,
    float *b )
{
	
	//int gid = j;
	int mask = Ns - 1; 
	
	//radix-2 fft	
	float v[4];
	//int idxS = gid;
	float xx;
	float yy;
	float angle = -2 *M_PI*(j & mask)/(Ns * 2) ;


	v[0] = a[2 * j ];
	v[1] = a[2 * j +1];
	
	v[2] = a[2 * j + N ];
	v[3] = a[2 * j + N +1];
	
	xx = v[2];
	yy = v[3];

	v[2] = xx * cos(angle)- yy* sin(angle); 
	v[3]=  xx * sin(angle) + yy * cos(angle);

	xx = v[0];
	yy = v[1];

	v[0] = xx + v[2];
	
	v[1] = yy + v[3];
	v[2] = xx - v[2];
	v[3] = yy - v[3];

	int idxD = (j / Ns) * Ns*2 + (j & mask);

	int k =  idxD << 1 ;
	

	b[k] = v[0];
	b[k+1] = v[1];
	b[k+2*Ns] = v[2];
	b[k+2*Ns+1] = v[3];



	
}



void main(int argc, char** argv)
{
	int  N  = pow(2,atoi(argv[1]));


	float * a = (float *) malloc(sizeof(float)* N* 2);
	float * b = (float *) malloc(sizeof(float)* N * 2);
	float * c = (float *) malloc(sizeof(float)* N * 2);
	float p = 2*M_PI ;	
	for (int i =0; i< N; i++)
	{
		a[2*i] = sin((p*i)/N);
		a[2*i+1] = 0;
		b[2*i] = 0;
		b[2*i+1] = 0;
	}
	int Ns;
	double elapsed;
	timestamp_type time1, time2;

	get_timestamp(&time1);

	float * d;
	float v[4];
		
	float xx;
	float yy;
	for (int j = 0; j<N/2; j++)
	{
		
		 
	
		//radix-2 fft	

		


		v[0] = a[2 * j ];
		v[1] = a[2 * j +1];
	
	
		v[2] = a[2 * j + N ];
		v[3] = a[2 * j + N +1];
	

		xx = v[0];
		yy = v[1];

		v[0] = xx + v[2];
		v[1] = yy + v[3];
		v[2] = xx - v[2];
		v[3] = yy - v[3];
		

		int k = j<<2 ;
	

		b[k] = v[0];
		b[k+1] = v[1];
		b[k+2] = v[2];
		b[k+3] = v[3];
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
	for( Ns = 2;Ns < N; Ns <<= 1 )
	{
		for (int j = 0; j<N/2; j++)
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


}


