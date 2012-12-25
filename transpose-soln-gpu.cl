//#pragma OPENCL EXTENSION cl_khr_fp64: enable

#define BLK 16

__kernel void transpose(
    __global float2 *a,
    __global float2 *b,
    int n,
    int option,
    float epsilon,
    float k,
    float s,
    long offset)
{

  local float2 l_a[16][16];
  long ig = get_group_id(0);
  long jg = get_group_id(1);
  long il = get_local_id(0);
  long jl = get_local_id(1);
  int ilsz = get_local_size(0);
  int jlsz = get_local_size(1);
  int nn=get_global_size(1);
  long i = ig * ilsz + il;
  long j = jg * ilsz + jl;
  int N =n;
  //l_a[il][jl] = a[i + n*j];
  if(0 == option)
    l_a[il][jl] = a[offset + i + n*j];
  else if(1 == option)
    {
	float2 pr = a[offset + i+ n*j];
	float a_i,a_j;
	if(i <= N/2)
	  a_i = i;
	else 
	  a_i = i - N;
	if(j <= N/2)
	  a_j = j;
	else 
	  a_j = j - N;
	float q = 2 + epsilon * k* pow((a_i*a_i + a_j*a_j),s+1)+2*k/epsilon * pow((a_i*a_i + a_j*a_j),s);
	//q= pow((a_i*a_i + a_j*a_j),s+1);
	pr /= q;
	l_a[il][jl] = pr;
    }
  else if(2 == option)
  {
      	float2 pr = a[offset + i+ n*j];
     	float a_i,a_j;
	if(i <= N/2)
	{
	  if(i == 0)
		a_i= 1;
          else
	  a_i = i;
	}
	else 
	  a_i = i - N;
	if(j <= N/2)
	  a_j = j;
	else 
	  a_j = j - N;

	float nlap =  pow((a_i*a_i + a_j*a_j),s);
	pr /= nlap;
	l_a[il][jl] = pr;
  }
  else if (3 == option)
  {
	float2 pr = a[offset + i+ n*j];
     	float a_i,a_j;
	if(i <= N/2)
	  a_i = i;
	else 
	  a_i = i - N;
	if(j <= N/2)
	  a_j = j;
	else 
	  a_j = j - N;

	float shar = - pow((a_i*a_i + a_j*a_j),s+1);
	pr *= shar;
	l_a[il][jl] = pr;
  }
  else if(4 == option)
  {
      	float2 pr = a[offset + i+ n*j];
     	float a_i,a_j;
	if(i <= N/2)
	  a_i = i;
	else 
	  a_i = i - N;
	if(j <= N/2)
	  a_j = j;
	else 
	  a_j = j - N;

	float nlap =  pow((a_i*a_i + a_j*a_j),s);
	pr *= nlap;
	l_a[il][jl] = pr;
  }
  else if(-1 == option)
  {
 	l_a[il][jl] = (float2 )(a[offset + i + n*j].x/(N*N),0);
  }
  else if(5 == option)
  {
	float2 pr = a[offset + i+ n*j];
     	float a_j;
	if(j < N/2)
	  a_j = j;
	else if(j == N/2)
	  a_j =0;
	else
	  a_j = j - N;


	
	pr = (float2)(-a_j*pr.y , a_j*pr.x);
	l_a[il][jl] = pr;
  }
  else if(6 == option)
  {
	float2 pr = a[offset + i+ n*j];
     	float a_i;
	if(i < N/2)
	  a_i = i;
	else if(i == N/2)
	  a_i =0;
	else
	  a_i = i - N;


	
	pr = (float2)(-a_i*pr.y , a_i*pr.x);
	l_a[il][jl] = pr;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  //b[offset + jg*jlsz + n*ig*ilsz + il + jl*n ] = l_a[jl][il] ;
  b[j+nn*i ] = l_a[il][jl] ;
}
