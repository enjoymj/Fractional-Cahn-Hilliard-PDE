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
  int jlsz = get_local_size(0);
  long i = ig * ilsz + il;
  long j = jg * ilsz + jl;
  int N =n;
  //l_a[il][jl] = a[i + n*j];
 
    l_a[il][jl] = a[offset + i + n*j];

  barrier(CLK_LOCAL_MEM_FENCE);
  b[offset + jg*jlsz + n*ig*ilsz + il + jl*n ] = l_a[jl][il] ;

}
