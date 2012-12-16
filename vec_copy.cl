//#pragma OPENCL EXTENSION cl_amd_fp64 : enable

__kernel void copy(
    __global const float *a,
    __global float *b, 
    long n)
{
  int gid = get_global_id(0);
  if (gid < n)
    b[gid] = a[gid];
}
