#pragma OPENCL EXTENSION cl_khr_fp64: enable

#define BLK 16

__kernel void transpose(
    __global double2 *a,
    __global double2 *b,
    int n)
{

  local double2 l_a[16][16];
  long ig = get_group_id(0);
  long jg = get_group_id(1);
  long il = get_local_id(0);
  long jl = get_local_id(1);

  long i = ig * BLK + il;
  long j = jg * BLK + jl;

  l_a[il][jl] = a[i + n*j];
  barrier(CLK_LOCAL_MEM_FENCE);
  b[jg*BLK + n*ig*BLK + il + jl*n ] = l_a[jl][il] ;

}
