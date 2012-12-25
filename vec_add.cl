//#pragma OPENCL EXTENSION cl_amd_fp64 : enable

__kernel void sum(
    __global const float16 *a,
    __global const float16 *b, 
    __global float16 *c,
    float a_mult,
    float b_mult,
    int n)
{
  int gid = get_global_id(0);
  if (gid < n)
    c[gid] = a[gid]*a_mult + b[gid] * b_mult;
}
