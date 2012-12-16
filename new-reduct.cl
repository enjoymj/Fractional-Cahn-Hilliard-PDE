__kernel void reduce2( 
	__global float2* a ,
	__global float2 * b,
	__global 
	int n,
	__local T∗ ldata)
	global T ∗g odata,
{
unsigned int lid = get local id (0);
unsigned int i = get global id (0);
ldata [ lid ] = (i < n) ? g idata [ i ] : 0;
barrier (CLK LOCAL MEM FENCE);
for (unsigned int s= get local size (0)/2; s>0; s>>=1)
{
if ( lid < s)
ldata [ lid ] += ldata[lid + s];
barrier (CLK LOCAL MEM FENCE);
}
if ( lid == 0) g odata[ get local size (0)] = ldata [0];
}

