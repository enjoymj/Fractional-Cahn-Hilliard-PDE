//#pragma OPENCL EXTENSION cl_amd_fp64 : enable

__kernel void zero(
    __global float *a,
    int n)
{
  int gid = get_global_id(0);
  if (gid < n)
    a[gid] = 0;
}
