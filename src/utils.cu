#include "../include/utils.h"
#include <stdio.h>
#include <cstdlib>

void HandleError(cudaError_t err, const char * file, int line)
{
	if (err != cudaSuccess)
	{
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}


int CudaGetThreadNum()
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	// cuda software arthitecture: thread ---> block ---> grid;
	printf("max thread num: %d\n", prop.maxThreadsPerBlock);
	printf("max grid dimension %d, %d, %d \n", 
		prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
	return prop.maxThreadsPerBlock;
}



