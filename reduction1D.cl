//#pragma OPENCL EXTENSION cl_amd_fp64 : enable

__kernel void reduction(
    __global float *a,
    int N)
{
  int gid = get_global_id(0);
  int lid = get_local_id(0);
  int grp = get_group_id(0);
  __local float l_a[128];
  
  l_a[lid] = a[gid];
  barrier(CLK_LOCAL_MEM_FENCE);
  
  size_t local_size = get_local_size(0);
  int count=0;
  while(local_size > 1)
  {
	barrier(CLK_LOCAL_MEM_FENCE);	
	if(lid < local_size/2)
	{
	     local_size /= 2;
             l_a[lid] += l_a[lid+ local_size];
	     //barrier(CLK_LOCAL_MEM_FENCE);
        }
	else
		local_size /= 2;
  }
  if(lid == 0)
     a[grp] = l_a[0];

}
