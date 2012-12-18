//#pragma OPENCL EXTENSION cl_khr_fp64: enable

#define BLK 16

__kernel void transpose_3D(
    __global float2 *a,
    __global float2 *b,
    int n,
    int option,
    float epsilon,
    float k,
    float s,
    int offset)
{

  local float2 l_a[16][16];
  long ig = get_group_id(0);
  long jg = get_group_id(1);
  long zg = get_group_id(2);
  long il = get_local_id(0);
  long jl = get_local_id(1);
  int ilsz = get_local_size(0);
  int jlsz = get_local_size(1);
  long z = get_global_size(2);
  long i = ig * ilsz + il;
  long j = jg * ilsz + jl;
  int N =n;
  //l_a[il][jl] = a[i + n*j];
  if(0 == option)
    l_a[il][jl] = a[z*zg + i + n*j];

  barrier(CLK_LOCAL_MEM_FENCE);
  b[z*zg + jg*jlsz + n*ig*ilsz + il + jl*n ] = l_a[jl][il] ;

}
