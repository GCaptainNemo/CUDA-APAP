#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


void HandleError(cudaError_t err, const char * file, int line);
int CudaGetThreadNum();


