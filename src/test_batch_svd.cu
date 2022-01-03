#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "../include/utils.h"

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <iostream>


//#include "Utilities.cuh"
//#include "TimingGPU.cuh"

#define FULLSVD
//#define PRINTRESULTS
#define gpuErrchk(err) (HandleError(err, __FILE__, __LINE__));
/********/
/* MAIN */
/********/
//#define EDITED_BY_WHY
int main() {

#ifdef EDITED_BY_WHY
	const int           M = 9;
	const int           N = 26;
	const int           lda = M;   // 行主序

#else
    const int           M           = 26;
    const int           N           = 9;
	const int           lda = M;   // 行主序

#endif
	//const int           lda         = N;

    //const int         numMatrices = 3;
    const int           numMatrices = 3;

    //TimingGPU timerGPU;

    // --- Setting the host matrix
    double *obj_mat = (double *)malloc(N * M * numMatrices * sizeof(double));
	//double(*obj_mat)[][9] = ((double *)[][9])malloc(lda * N * numMatrices * sizeof(double));
	double _linshi_0[][9] = {- 18.181494, -59.013740, 0.100000, 0.000000, 0.000000, 0.000000, -10745.918129, -34879.244270, -59.103602,
0.000000, 0.000000, 0.000000, 18.181494, 59.013740, 0.100000, -7295.509484, -23679.862853, -40.126017,
18.181494, 59.013740, 0.100000, 0.000000, 0.000000, 0.000000, -10745.918129, -34879.244270, -59.103602,
0.000000, 0.000000, 0.000000, 18.181494, 59.013740, 0.100000, -7295.509484, -23679.862853, -40.126017,
51.083848, 62.305158, 0.100000, 0.000000, 0.000000, 0.000000, -34595.019266, -42194.319379, -67.722035,
0.000000, 0.000000, 0.000000, 51.083848, 62.305158, 0.100000, -20924.223749, -25520.533193, -40.960547,
52.879639, 75.852345, 0.100000, 0.000000, 0.000000, 0.000000, -36099.328663, -51782.100772, -68.266975,
0.000000, 0.000000, 0.000000, 52.879639, 75.852345, 0.100000, -23570.803476, -33810.759879, -44.574442,
54.732325, 66.605696, 0.100000, 0.000000, 0.000000, 0.000000, -37632.947436, -45796.860057, -68.758174,
0.000000, 0.000000, 0.000000, 54.732325, 66.605696, 0.100000, -23031.606593, -28027.975418, -42.080448,
57.554035, 33.898780, 0.100000, 0.000000, 0.000000, 0.000000, -40065.325597, -23598.095664, -69.613410,
0.000000, 0.000000, 0.000000, 57.554035, 33.898780, 0.100000, -19281.820600, -11356.809544, -33.502118,
65.578846, 42.619486, 0.100000, 0.000000, 0.000000, 0.000000, -47118.581952, -30622.219206, -71.850276,
0.000000, 0.000000, 0.000000, 65.578846, 42.619486, 0.100000, -23451.626912, -15241.137727, -35.760962,
65.578846, 42.619486, 0.100000, 0.000000, 0.000000, 0.000000, -47118.581952, -30622.219206, -71.850276,
0.000000, 0.000000, 0.000000, 65.578846, 42.619486, 0.100000, -23451.626912, -15241.137727, -35.760962,
72.897993, 77.164479, 0.100000, 0.000000, 0.000000, 0.000000, -53827.788302, -56978.157099, -73.839875,
0.000000, 0.000000, 0.000000, 72.897993, 77.164479, 0.100000, -32827.997364, -34749.316143, -45.032789,
73.008491, 47.666270, 0.100000, 0.000000, 0.000000, 0.000000, -53958.663304, -35228.891150, -73.907380,
0.000000, 0.000000, 0.000000, 73.008491, 47.666270, 0.100000, -27062.797278, -17668.939326, -37.068015,
73.001753, 71.172908, 0.100000, 0.000000, 0.000000, 0.000000, -53968.957054, -52616.919534, -73.928303,
0.000000, 0.000000, 0.000000, 73.001753, 71.172908, 0.100000, -31673.147347, -30879.669210, -43.386832,
73.001753, 71.172908, 0.100000, 0.000000, 0.000000, 0.000000, -53968.957054, -52616.919534, -73.928303,
0.000000, 0.000000, 0.000000, 73.001753, 71.172908, 0.100000, -31673.147347, -30879.669210, -43.386832,
73.001753, 71.172908, 0.100000, 0.000000, 0.000000, 0.000000, -53968.957054, -52616.919534, -73.928303,
0.000000, 0.000000, 0.000000, 73.001753, 71.172908, 0.100000, -31673.147347, -30879.669210, -43.386832 };

	double _linshi[][9] = { 18.181494, 59.013740, 0.100000, 0.000000, 0.000000, 0.000000, -10745.918129, -34879.244270, -59.103602,
0.000000, 0.000000, 0.000000, 18.181494, 59.013740, 0.100000, -7295.509484, -23679.862853, -40.126017,
18.181494, 59.013740, 0.100000, 0.000000, 0.000000, 0.000000, -10745.918129, -34879.244270, -59.103602,
0.000000, 0.000000, 0.000000, 18.181494, 59.013740, 0.100000, -7295.509484, -23679.862853, -40.126017,
51.083848, 62.305158, 0.100000, 0.000000, 0.000000, 0.000000, -34595.019266, -42194.319379, -67.722035,
0.000000, 0.000000, 0.000000, 51.083848, 62.305158, 0.100000, -20924.223749, -25520.533193, -40.960547,
52.879639, 75.852345, 0.100000, 0.000000, 0.000000, 0.000000, -36099.328663, -51782.100772, -68.266975,
0.000000, 0.000000, 0.000000, 52.879639, 75.852345, 0.100000, -23570.803476, -33810.759879, -44.574442,
54.732325, 66.605696, 0.100000, 0.000000, 0.000000, 0.000000, -37632.947436, -45796.860057, -68.758174,
0.000000, 0.000000, 0.000000, 54.732325, 66.605696, 0.100000, -23031.606593, -28027.975418, -42.080448,
57.554035, 33.898780, 0.100000, 0.000000, 0.000000, 0.000000, -40065.325597, -23598.095664, -69.613410,
0.000000, 0.000000, 0.000000, 57.554035, 33.898780, 0.100000, -19281.820600, -11356.809544, -33.502118,
65.578846, 42.619486, 0.100000, 0.000000, 0.000000, 0.000000, -47118.581952, -30622.219206, -71.850276,
0.000000, 0.000000, 0.000000, 65.578846, 42.619486, 0.100000, -23451.626912, -15241.137727, -35.760962,
65.578846, 42.619486, 0.100000, 0.000000, 0.000000, 0.000000, -47118.581952, -30622.219206, -71.850276,
0.000000, 0.000000, 0.000000, 65.578846, 42.619486, 0.100000, -23451.626912, -15241.137727, -35.760962,
72.897993, 77.164479, 0.100000, 0.000000, 0.000000, 0.000000, -53827.788302, -56978.157099, -73.839875,
0.000000, 0.000000, 0.000000, 72.897993, 77.164479, 0.100000, -32827.997364, -34749.316143, -45.032789,
73.008491, 47.666270, 0.100000, 0.000000, 0.000000, 0.000000, -53958.663304, -35228.891150, -73.907380,
0.000000, 0.000000, 0.000000, 73.008491, 47.666270, 0.100000, -27062.797278, -17668.939326, -37.068015,
73.001753, 71.172908, 0.100000, 0.000000, 0.000000, 0.000000, -53968.957054, -52616.919534, -73.928303,
0.000000, 0.000000, 0.000000, 73.001753, 71.172908, 0.100000, -31673.147347, -30879.669210, -43.386832,
73.001753, 71.172908, 0.100000, 0.000000, 0.000000, 0.000000, -53968.957054, -52616.919534, -73.928303,
0.000000, 0.000000, 0.000000, 73.001753, 71.172908, 0.100000, -31673.147347, -30879.669210, -43.386832,
73.001753, 71.172908, 0.100000, 0.000000, 0.000000, 0.000000, -53968.957054, -52616.919534, -73.928303,
0.000000, 0.000000, 0.000000, 73.001753, 71.172908, 0.100000, -31673.147347, -30879.669210, -43.386832 };
#ifdef EDITED_BY_WHY
	for (unsigned int k = 0; k < numMatrices; k++) {
		for (unsigned int i = 0; i < N; i++) {
			for (unsigned int j = 0; j < M; j++) 
			{
				if (k % 2 == 0) {
					obj_mat[k * M * N + i * M + j] = _linshi[i][j];
				}
				else {
					obj_mat[k * M * N + i * M + j] = _linshi_0[i][j];
				}
			}
		}
	}
	for (unsigned int i = 0; i < N; i++) {
		for (unsigned int j = 0; j < M; j++) {
			printf("%f, ", obj_mat[i * M + j]);
		}
		printf("\n");
	}

#else
	for (unsigned int k = 0; k < numMatrices; k++) {
		for (unsigned int i = 0; i < M; i++) {
			for (unsigned int j = 0; j < N; j++) {
				if (k % 2 == 0) {
					//obj_mat[k * M * N + j * M + i] = (1. / (k + 1)) * (i + j * j) * (i + j);
					obj_mat[k * M * N + j * M + i] = _linshi[i][j];

						//obj_mat[(k * M + i) * N + j] = _linshi[i][j];

					//printf("%d %d %f\n", i, j, obj_mat[j*M + i]);
				}
				else {
					obj_mat[k * M * N + j * M + i] = _linshi_0[i][j];
					//obj_mat[(k * M + i) * N + j] = _linshi_0[i][j];

				}
			}
		}
	}
	for (unsigned int i = 0; i < M; i++) {
		for (unsigned int j = 0; j < N; j++) {
			printf("%f, ", obj_mat[j * M + i]);
		}
		printf("\n");
	}
#endif

	


    // --- Setting the device matrix and moving the host matrix to the device
    double *gpu_obj_mat;         
	cudaMalloc(&gpu_obj_mat, M * N * numMatrices * sizeof(double));
    cudaMemcpy(gpu_obj_mat, obj_mat, M * N * numMatrices * sizeof(double), cudaMemcpyHostToDevice);

    // --- host side SVD results space
    double *h_S = (double *)malloc(N * numMatrices * sizeof(double));
    double *h_U = NULL;
    double *h_V = NULL;
#ifdef FULLSVD
            h_U = (double *)malloc(M * lda * numMatrices * sizeof(double));
            h_V = (double *)malloc(N * lda * numMatrices * sizeof(double));
#endif

    // --- device side SVD workspace and matrices
    int work_size = 0;

    int *devInfo;        cudaMalloc(&devInfo, sizeof(int));
    double *d_S;         cudaMalloc(&d_S, N * numMatrices * sizeof(double));
    double *d_U = NULL;
    double *d_V = NULL;
#ifdef FULLSVD
                         gpuErrchk(cudaMalloc(&d_U, M * lda * numMatrices * sizeof(double)));
                         gpuErrchk(cudaMalloc(&d_V, N * lda * numMatrices * sizeof(double)));
#endif

    double *d_work = NULL; /* devie workspace for gesvdj */
    int devInfo_h = 0; /* host copy of error devInfo_h */

    // --- Parameters configuration of Jacobi-based SVD
    const double            tol             = 1.e-7;
    const int               maxSweeps       = 15;
          cusolverEigMode_t jobz;                                   // --- CUSOLVER_EIG_MODE_VECTOR - Compute eigenvectors; CUSOLVER_EIG_MODE_NOVECTOR - Compute singular values only
#ifdef FULLSVD
        jobz = CUSOLVER_EIG_MODE_VECTOR;
#else
        jobz = CUSOLVER_EIG_MODE_NOVECTOR;    // only calculate sigular values
#endif
    const int               econ            = 0;                            // --- econ = 1 for economy size 

    // --- Numerical result parameters of gesvdj 
    double                  residual        = 0;
    int                     executedSweeps  = 0;

    // --- CUDA solver initialization
    cusolverDnHandle_t solver_handle = NULL;
    cusolverDnCreate(&solver_handle);

    // --- Configuration of gesvdj
    gesvdjInfo_t gesvdj_params = NULL;
    cusolverDnCreateGesvdjInfo(&gesvdj_params);

    // --- Set the computation tolerance, since the default tolerance is machine precision
    cusolverDnXgesvdjSetTolerance(gesvdj_params, tol);

    // --- Set the maximum number of sweeps, since the default value of max. sweeps is 100
    cusolverDnXgesvdjSetMaxSweeps(gesvdj_params, maxSweeps);

    // --- Query the SVD workspace 
    cusolverDnDgesvdjBatched_bufferSize(
        solver_handle,
        jobz,                                       // --- Compute the singular vectors or not
        M,                                          // --- Nubmer of rows of A, 0 <= M
        N,                                          // --- Number of columns of A, 0 <= N 
        gpu_obj_mat,                                        // --- M x N
        lda,                                        // --- Leading dimension of A
        d_S,                                        // --- Square matrix of size min(M, N) x min(M, N)
        d_U,                                        // --- M x M if econ = 0, M x min(M, N) if econ = 1
        lda,                                        // --- Leading dimension of U, ldu >= max(1, M)
        d_V,                                        // --- N x N if econ = 0, N x min(M,N) if econ = 1
        lda,                                        // --- Leading dimension of V, ldv >= max(1, N)
        &work_size,
        gesvdj_params,
        numMatrices);

    cudaMalloc(&d_work, sizeof(double) * work_size);

    // --- Compute SVD
    //timerGPU.StartCounter();
    cusolverDnDgesvdjBatched(
        solver_handle,
        jobz,                                       // --- Compute the singular vectors or not
        M,                                          // --- Number of rows of A, 0 <= M
        N,                                          // --- Number of columns of A, 0 <= N 
        gpu_obj_mat,                                        // --- M x N
        lda,                                        // --- Leading dimension of A
        d_S,                                        // --- Square matrix of size min(M, N) x min(M, N)
        d_U,                                        // --- M x M if econ = 0, M x min(M, N) if econ = 1
        lda,                                        // --- Leading dimension of U, ldu >= max(1, M)
        d_V,                                        // --- N x N if econ = 0, N x min(M, N) if econ = 1
        lda,                                        // --- Leading dimension of V, ldv >= max(1, N)
        d_work,
        work_size,
        devInfo,
        gesvdj_params,
        numMatrices);
	cudaDeviceSynchronize();
    //printf("Calculation of the singular values only: %f ms\n\n", timerGPU.GetCounter());

    gpuErrchk(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_S, d_S, sizeof(double) *       N * numMatrices, cudaMemcpyDeviceToHost));
	std::cout << "h_S = " << std::endl;
	for (int k = 0; k < numMatrices; ++k) {
		for (int i = 0; i < N; ++i) {
			std::cout << h_S[k * N + i] << ", ";
		}
		std::cout << std::endl;
	}
#ifdef FULLSVD
    cudaMemcpy(h_U, d_U, sizeof(double) * M * lda * numMatrices, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_V, d_V, sizeof(double) * N * lda * numMatrices, cudaMemcpyDeviceToHost);
	
#endif

#ifdef PRINTRESULTS
    printf("SINGULAR VALUES \n");
    printf("_______________ \n");
    for (int k = 0; k < numMatrices; k++) {
        for (int p = 0; p < N; p++)
            printf("Matrix nr. %d; SV nr. %d; Value = %f\n", k, p, h_S[k * N + p]);
        printf("\n");
    }
#endif
#ifdef FULLSVD
    /*printf("SINGULAR VECTORS U \n");
    printf("__________________ \n");
    for (int k = 0; k < numMatrices; k++) {
        for (int q = 0; q < (1 - econ) * M + econ * min(M, N); q++)
            for (int p = 0; p < M; p++)
                printf("Matrix nr. %d; U nr. %d; Value = %f\n", k, p, h_U[((1 - econ) * M + econ * min(M, N)) * M * k + q * M + p]);
        printf("\n");
    }*/

    printf("SINGULAR VECTORS V \n"); // V is Nxm Matrix, NXN is vt, and N x (M-N) =null
    printf("__________________ \n");
	if (econ == 0) {
		for (int k = 0; k < numMatrices; k++)
		{
			for (int q = 0; q < N; q++) {
				for (int p = 0; p < N; p++) {
					//printf("%f, ", h_V[N * M * k + q * M + p]);
					printf("%f, ", h_V[N * M * k + q * M + p]);

				}
				printf("\n");
			}
			printf("---------\n");
		}
	}
  //  for (int k = 0; k < numMatrices; k++) {
		//for (int q = 0; q < (1 - econ) * N + econ * min(M, N); q++) {
		//	for (int p = 0; p < M; p++)
		//		printf("%f, ", h_V[((1 - econ) * N + econ * min(M, N)) * N * k + q * N + p]);
		//	//printf("Matrix nr. %d; V nr. %d; Value = %f\n", k, p, h_V[((1 - econ) * N + econ * min(M, N)) * N * k + q * N + p]);
		//	printf("\n");
		//}
		//printf("----------------------------------\n");
  //  }
#endif

    if (0 == devInfo_h){
        printf("gesvdj converges \n");
    }
    else if (0 > devInfo_h){
        printf("%d-th parameter is wrong \n", -devInfo_h);
        exit(1);
    }
    else{
        printf("WARNING: devInfo_h = %d : gesvdj does not converge \n", devInfo_h);
    }

    // --- Free resources
    if (gpu_obj_mat) gpuErrchk(cudaFree(gpu_obj_mat));
    if (d_S) gpuErrchk(cudaFree(d_S));
#ifdef FULLSVD
    if (d_U) gpuErrchk(cudaFree(d_U));
    if (d_V) gpuErrchk(cudaFree(d_V));
#endif
    if (devInfo) gpuErrchk(cudaFree(devInfo));
    if (d_work) gpuErrchk(cudaFree(d_work));
    if (solver_handle) cusolverDnDestroy(solver_handle);
    if (gesvdj_params) cusolverDnDestroyGesvdjInfo(gesvdj_params);

    gpuErrchk(cudaDeviceReset());

    return 0;
}