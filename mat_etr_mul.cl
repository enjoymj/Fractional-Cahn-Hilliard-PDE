//#pragma OPENCL EXTENSION cl_amd_fp64 : enable

__kernel void mult(
    __global const float *a,
    __global const float *b, 
    __global float *c,
    long n)
{
  int gid = get_global_id(0);
  if (gid < n)
    c[gid] = a[gid] * b[gid];
}
