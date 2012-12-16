//#pragma OPENCL EXTENSION cl_amd_fp64 : enable

__kernel void fft1D(
    __global const float *a,
    __global float *b, 
    int N,
    int Ns)
{
  int gid = get_global_id(0);
 
}
