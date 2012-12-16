//#pragma OPENCL EXTENSION cl_amd_fp64 : enable

__kernel void reduction_mult(
    __global float2 *a,
    __global float2 *b,
    __global float *c,
    int N)
{
  int gid = get_global_id(0);
  int lid = get_local_id(0);
  int grp = get_group_id(0);
  __local float l_a[128];
  
  l_a[lid] = a[gid].x * b[gid].x;
  barrier(CLK_LOCAL_MEM_FENCE);
  
  size_t size = get_local_size(0);

  while(size >1)
  {
	barrier(CLK_LOCAL_MEM_FENCE);	
	if(lid < size/2)
	{
	     size /= 2;
             l_a[lid] += l_a[lid+ size];
        }
	else
	{size/=2;
	
	}

  }
		
  if(lid == 0)
     {
	
	c[grp ] = l_a[0];
	
	}

}
