#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <cusolverDn.h>
#include <string>
#include "match.h"
#include "dlt.h"
#include "utils.h"
#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include <vector>

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__));
//#define DEBUG

using namespace std;
constexpr int threads_x_per_block = 16;
constexpr int threads_y_per_block = 16;
constexpr float gamma = 0.1f; // Normalizer for Moving DLT. (0.0015 - 0.1 are usually good numbers).
constexpr float std_variance = 8.5f; // Bandwidth for Moving DLT. (Between 8 - 12 are good numbers).




__device__ float eu_dist(float src_x, float src_y, float dst_x, float dst_y) {
	return sqrt((src_x - dst_x) * (src_x - dst_x) + (src_y - dst_y) * (src_y - dst_y));
};


__global__ void kernel_get_weighted_A(float(*A)[9], float (*dst_img_match_pts)[2], float (* obj_pos)[2], 
	double (*weighted_A)[9],
	int match_num, int modify_match_num, int obj_pos_rows, int obj_pos_cols)
{
	// output weighted_A: num_x * num_y * (match_pts * 2 ) * 9 

	int thread_id_x = threadIdx.x;
	int block_id_x = blockIdx.x;
	int obj_pos_x = block_id_x * blockDim.x + thread_id_x;
	// 
	int thread_id_y = threadIdx.y;
	int block_id_y = blockIdx.y;
	int obj_pos_y = block_id_y * blockDim.y + thread_id_y;
	if (obj_pos_y >= obj_pos_rows || obj_pos_x >= obj_pos_cols)
	{
		return;
	}
	int pos_id = obj_pos_y * obj_pos_cols + obj_pos_x;
	double inv_sigma = 1 / std_variance / std_variance;
	if (match_num == modify_match_num) {
		for (int i = 0; i < match_num; ++i) {
			double dist = eu_dist(obj_pos[pos_id][0], obj_pos[pos_id][1],
				dst_img_match_pts[i][0], dst_img_match_pts[i][1]);
			double weight = (double)max(expf(-dist * inv_sigma), gamma);
			for (int j = 0; j < 9; ++j)
			{
				weighted_A[pos_id * (2 * match_num) + i * 2][j] = weight * double(A[i * 2][j]);
				weighted_A[pos_id * (2 * match_num) + (i * 2 + 1)][j] = weight * double(A[i * 2 + 1][j]);
			}
		}
	}
	else {
		// match_num > modify_match_num
		// construct heap data structure to solve the topK problem(O(NlogK)).
		double *min_heap_weight = new double[modify_match_num];
		int *min_heap_index = new int[modify_match_num];
		for (int i = 0; i < match_num; ++i) 
		{
			double dist =
				eu_dist(obj_pos[pos_id][0], obj_pos[pos_id][1], dst_img_match_pts[i][0], dst_img_match_pts[i][1]);
			double weight = (double)max(expf(-dist * inv_sigma), gamma);
			if (i < modify_match_num) {
				// construct min_heap
				min_heap_weight[i] = weight;
				min_heap_index[i] = i;
				int cur_index = i;
				while (true){
					if (cur_index == 0) { break; }
					int father_index = (cur_index - 1) / 2;
					if (min_heap_weight[father_index] > min_heap_weight[cur_index]) {
						double linshi_w = min_heap_weight[father_index];
						min_heap_weight[father_index] = min_heap_weight[cur_index];
						min_heap_weight[cur_index] = linshi_w;
						// 
						int linshi_index = min_heap_index[father_index];
						min_heap_index[father_index] = min_heap_index[cur_index];
						min_heap_index[cur_index] = linshi_index;
						// shift up
						cur_index = father_index;
					}
					else {
						break;
					}
				}
			
			}
			else {
				if (weight < min_heap_weight[0]) { continue; }
				else {
					min_heap_weight[0] = weight;
					min_heap_index[0] = i;
					int cur_index = 0;
					while (true) {
						int left_son_index = cur_index * 2 + 1;
						int right_son_index = cur_index * 2 + 2;
						double left_son_val = (left_son_index < modify_match_num? min_heap_weight[left_son_index] : DBL_MAX);
						double right_son_val = (right_son_index < modify_match_num ? min_heap_weight[right_son_index]: DBL_MAX);
						if (min_heap_weight[cur_index] <= left_son_val
							&& min_heap_weight[cur_index] <= right_son_val) {
							break;
						}
						else if (left_son_val < right_son_val) {
							min_heap_weight[left_son_index] = min_heap_weight[cur_index];
							min_heap_weight[cur_index] = left_son_val;

							min_heap_index[left_son_index] = min_heap_index[cur_index];
							min_heap_index[cur_index] = left_son_index;
							cur_index = left_son_index;
						}
						else{
							min_heap_weight[right_son_index] = min_heap_weight[cur_index];
							min_heap_weight[cur_index] = right_son_val;

							min_heap_index[right_son_index] = min_heap_index[cur_index];
							min_heap_index[cur_index] = right_son_index;
							cur_index = right_son_index;
						}

					}
				}
			}
		};
		__syncthreads();
		for (int i = 0; i < modify_match_num; ++i) {
			double weight = min_heap_weight[i];
			int index = min_heap_index[i];
			for (int j = 0; j < 9; ++j)
			{
				//if (pos_id == 0) {
				//printf("weight * double(A[index * 2][j] = %f\n", weight * double(A[index * 2][j]));
				//}
				//weighted_A[pos_id * (2 * modify_match_num) + i * 2][j] = 1.0;
				//weighted_A[pos_id * (2 * modify_match_num) + (i * 2 + 1)][j] = 1.0f;
				//A[index * 2][j] = 1.0;
				//A[index * 2 + 1][j] = 1.0;

				weighted_A[pos_id * (2 * modify_match_num) + i * 2][j] = weight * double(A[index * 2][j]);
				weighted_A[pos_id * (2 * modify_match_num) + (i * 2 + 1)][j] = weight * double(A[index * 2 + 1][j]);
			}
		}
		#ifdef DEBUG
		if (pos_id == 180) {
			printf("min_heap_weight = \n");
			for (int i = 0; i < modify_match_num; ++i) {
				printf("%f, ", min_heap_weight[i]);
			}
			printf("\n");
			printf("min_heap_index = \n");
			for (int i = 0; i < modify_match_num; ++i) {
				printf("%d, ", min_heap_index[i]);
			}
			printf("\n");

		}
		#endif
		if (min_heap_weight) { delete[] min_heap_weight; min_heap_weight = nullptr; }
		if (min_heap_index) { delete[] min_heap_index; min_heap_index = nullptr; }

	}
//#ifdef DEBUG
//	if (pos_id == 180) {
//		printf("gpu_weighted_A = \n");
//		for (int i = 0; i < 2 * modify_match_num; ++i) {
//			for (int j = 0; j < 9; ++j)
//			{
//				printf("%f, ", weighted_A[i][j]);
//			}
//			printf("\n");
//		}
//
//		printf("gpu_A = \n");
//		for (int i = 0; i < 2 * match_num; ++i) {
//			for (int j = 0; j < 9; ++j)
//			{
//				printf("%f, ", A[i][j]);
//			}
//			printf("\n");
//		}
//
//	}
//#endif

}	

__global__ void kernel_transpose(double(*in_mat)[9], double *out_mat, 
	const int match_num, const int mat_num_x, const int mat_num_y)
{
	// ////////////////////////////////////////////////////////////////////////////
	// input in_mat: mat_num_y * mat_num_x * (2 * match_num) * 9
	// output out_mat: mat_num_y * mat_num_x * 9 * (2 * match_num) 
	// ////////////////////////////////////////////////////////////////////////////
	extern __shared__ double block_mat[][9]; // shared memory dynamic allocation
	int thread_id_x = threadIdx.x;
	int thread_id_y = threadIdx.y;
	int block_id_x = blockIdx.x;
	int block_id_y = blockIdx.y;
	if (block_id_x >= mat_num_x || block_id_y >= mat_num_y 
		|| thread_id_x >= 9 || thread_id_y >= 2 * match_num) {
		return;
	}
	int index = (block_id_y * mat_num_x + block_id_x) * (2 * match_num) + thread_id_y;
	block_mat[thread_id_y][thread_id_x] = in_mat[index][thread_id_x];
	__syncthreads();
	int out_index = (block_id_y * mat_num_x + block_id_x) * (2 * match_num) * 9;
	out_mat[out_index + thread_id_x * 2 * match_num + thread_id_y] = block_mat[thread_id_y][thread_id_x];


}

__global__ void kernel_APAP(float *src_img,
	float *transformed_img, float *optical_flow, float (*homo)[3][3], 
	int homo_row, int homo_col,
	int dst_img_rows, int dst_img_cols, int offset_x, int offset_y, int src_img_rows, int src_img_cols)
{
	// /////////////////////////////////////////////////////////////////////////////////////////
	// input:
	// src_img(src_img_row * src_img_col * 1)

	// output:
	// transformed_img(dst_img_row * dst_img_col * 1): transform src to dst
	// optical_flow(dst_img_row * dst_img_col * 2)

	// USE APAP Weighted Direct Linear Transform algorithm to calculate "dst image-to-src image" optical flow field
	// and transform src img(gray) to dst img(gray)
	// //////////////////////////////////////////////////////////////////////////////////////////

	__shared__ float block_homography[3][3];
	int thread_id_x = threadIdx.x;
	int block_id_x = blockIdx.x;
	int thread_id_y = threadIdx.y;
	int block_id_y = blockIdx.y;
	
	int pixel_id_x = block_id_x * blockDim.x + thread_id_x;
	int pixel_id_y = block_id_y * blockDim.y + thread_id_y;

	if (pixel_id_y >= dst_img_rows || pixel_id_x >=  dst_img_cols)
	{
		return;
	}
	int img_row = pixel_id_y;
	int img_col = pixel_id_x;
	
	transformed_img[img_row * dst_img_cols + img_col] = 0.0f;

	if (thread_id_x == 0 && thread_id_y == 0)
	{
		int last_singular_index = (block_id_y * homo_col + block_id_x);
		for (int i = 0; i < 3; ++i) {

			for (int j = 0; j < 3; ++j) 
			{
				block_homography[i][j] = (float)homo[last_singular_index][i][j];
			}
		}
	}
	__syncthreads();   //  synchronize thread in a block

	// //////////////////////////////////////////////////////
	// calculate obj pts
	// //////////////////////////////////////////////////////
	float x = (float)pixel_id_x - (float) offset_x;
	float y = (float)pixel_id_y - (float) offset_y;
	// [h11 h12 h13] [x]   [x']
	// [h21 h22 h23] [y] = [y']
	// [h31 h32 h33] [1]   [s']
	float denominator = block_homography[2][0] * x + block_homography[2][1] * y + block_homography[2][2] * 1.0 + 1e-8;
	float obj_x = (block_homography[0][0] * x + block_homography[0][1] * y + block_homography[0][2] * 1.0) / denominator;
	float obj_y = (block_homography[1][0] * x + block_homography[1][1] * y + block_homography[1][2] * 1.0) / denominator;
	int obj_row = int(obj_y);
	int obj_col = int(obj_x);
	if (obj_y < 0 || obj_y >= src_img_rows - 1 || obj_x < 0 || obj_x >= src_img_cols - 1) {
		return;
	}
	

	// //////////////////////////////////////////////////////
	// calculate optical flow and transformed img
	// //////////////////////////////////////////////////////
	optical_flow[2 * (img_row * dst_img_cols + img_col)] = obj_x - x;
	optical_flow[2 * (img_row * dst_img_cols + img_col) + 1] = obj_y - y;

	// bilinear interpolation
	float delta_x = obj_x - float(obj_col);
	float delta_y = obj_y - float(obj_row);
	float interp_pixel;
	interp_pixel = delta_x * delta_y * src_img[(obj_row + 1)* src_img_cols + obj_col + 1] +
		(1 - delta_x) * (1 - delta_y)* src_img[obj_row * src_img_cols + obj_col] +
		(1 - delta_x) * delta_y * src_img[(obj_row + 1) * src_img_cols + obj_col] +
		delta_x * (1 - delta_y)* src_img[obj_row * src_img_cols + obj_col + 1];
	transformed_img[img_row * dst_img_cols + img_col] = interp_pixel;

}


void cuda_get_weighted_A(const int &patch_size_x, const int &patch_size_y,
	const int &dst_img_row, const int &dst_img_col, const int &offset_x, const int &offset_y,
	const float dst_match_pts[][2], const int &match_num, float(*A)[9],
	int &x_num, int &y_num, double(**cpu_weighted_A)[9])
{
	cudaSetDevice(0);
	float origin_x = float(patch_size_x - 1.0f) / 2.0f - (float)offset_x;
	float origin_y = float(patch_size_y - 1.0f) / 2.0f - (float)offset_y;
	x_num = (dst_img_col - 1) / patch_size_x + 1;
	y_num = (dst_img_row - 1) / patch_size_y + 1;
	float(*obj_pos)[2] = new float[x_num * y_num][2];

	// //////////////////////////////////////////////////////
	for (int i = 0; i < y_num; i++)
	{
		for (int j = 0; j < x_num; j++) {
			obj_pos[i * x_num + j][0] = origin_x + j * patch_size_x;
			obj_pos[i * x_num + j][1] = origin_y + i * patch_size_y;
			//std::cout << "(" << obj_pos[i * x_num + j][0] << "," << obj_pos[i * x_num + j][1] << ")";
		}
		//std::cout << std::endl;
	}

	// ///////////////////////////////////////////////////////
	float(*gpu_dst_match_pts)[2] = nullptr;
	float(*gpu_A)[9] = nullptr;
	float(*gpu_obj_pos)[2] = nullptr;
	double(*gpu_weighted_A)[9] = nullptr;
	int numMatrix = x_num * y_num;
	size_t match_pts_size = match_num * 2 * sizeof(float);
	size_t A_size = (match_num * 2) * 9 * sizeof(float);
	size_t obj_pos_size = numMatrix * 2 * sizeof(float);
	int modify_match_num = MIN(match_num, 16);
	size_t weighted_A_size = numMatrix * (2 * modify_match_num) * 9 * sizeof(double);

	HANDLE_ERROR(cudaMalloc((void **)& gpu_dst_match_pts, match_pts_size));
	HANDLE_ERROR(cudaMalloc((void **)& gpu_A, A_size));
	HANDLE_ERROR(cudaMalloc((void **)& gpu_obj_pos, obj_pos_size));
	HANDLE_ERROR(cudaMalloc((void **)& gpu_weighted_A, weighted_A_size));

	// memory copy kernel and src_gray from host to device
	HANDLE_ERROR(cudaMemcpy(gpu_dst_match_pts, dst_match_pts, match_pts_size, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(gpu_A, A, A_size, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(gpu_obj_pos, obj_pos, obj_pos_size, cudaMemcpyHostToDevice));

	// //////////////////////////////////////////////////////////////////
	// invoke kernel function
	// //////////////////////////////////////////////////////////////////
	int block_dim_x = 32;
	int block_dim_y = 32;
	int grid_dim_x = (x_num - 1) / block_dim_x + 1;
	int grid_dim_y = (y_num - 1) / block_dim_y + 1;
	dim3 grid_size(grid_dim_x, grid_dim_y, 1);
	dim3 block_size(block_dim_x, block_dim_y, 1);
	kernel_get_weighted_A <<<grid_size, block_size>>>(gpu_A, gpu_dst_match_pts, 
		gpu_obj_pos, gpu_weighted_A,
		match_num, modify_match_num, y_num, x_num);
	cudaDeviceSynchronize();
	// ///////////////////////////////////////////////////////////////////
	// get result
	// ///////////////////////////////////////////////////////////////////
	*cpu_weighted_A = new double[numMatrix * (2 * modify_match_num)][9];
	HANDLE_ERROR(cudaMemcpy((*cpu_weighted_A), gpu_weighted_A, weighted_A_size, cudaMemcpyDeviceToHost));

	// ///////////////////////////////////////////////////////////////////
	// Release Memory
	// ///////////////////////////////////////////////////////////////////
	HANDLE_ERROR(cudaFree(gpu_dst_match_pts));
	HANDLE_ERROR(cudaFree(gpu_obj_pos));

	HANDLE_ERROR(cudaFree(gpu_A));
	HANDLE_ERROR(cudaFree(gpu_weighted_A));

	gpu_dst_match_pts = nullptr;
	gpu_A = nullptr;
	gpu_obj_pos = nullptr;
	gpu_weighted_A = nullptr;
#ifdef DEBUG
	printf("\n A = \n");
	for (int i = 0; i < 2 * match_num; ++i)
	{
		for (int j = 0; j < 9; ++j) {
			printf("%f, ", A[i][j]);
		}
		printf("\n");
	}
	printf("\n weighted_A = \n");
	for (int i = 0; i < 2 * modify_match_num; ++i)
	{
		for (int j = 0; j < 9; ++j) {
			printf("%f, ", (*cpu_weighted_A)[i][j]);
		}
		printf("\n");
	}
#endif
}


void cuda_transpose(double(*cpu_weighted_A)[9], double **out_mat,
	int mat_num_x, int mat_num_y, int match_num)
{
	double(*gpu_weighted_A)[9] = nullptr;
	double *gpu_out_mat = nullptr;

	size_t gpu_weighted_A_size = mat_num_x * mat_num_y * 9 * (2 * match_num) * sizeof(double);

	HANDLE_ERROR(cudaMalloc((void **)& gpu_weighted_A, gpu_weighted_A_size));
	HANDLE_ERROR(cudaMalloc((void **)& gpu_out_mat, gpu_weighted_A_size));


	// memory copy kernel and src_gray from host to device
	HANDLE_ERROR(cudaMemcpy(gpu_weighted_A, cpu_weighted_A, gpu_weighted_A_size, cudaMemcpyHostToDevice));

	int block_dim_x = 9;
	int block_dim_y = match_num * 2;
	int grid_dim_x = mat_num_x;
	int grid_dim_y = mat_num_y;

	dim3 grid_size(grid_dim_x, grid_dim_y, 1);
	dim3 block_size(block_dim_x, block_dim_y, 1);
	size_t shared_memory_byte = 2 * match_num * 9 * sizeof(double);
	kernel_transpose << < grid_size, block_size, shared_memory_byte >> > (gpu_weighted_A, gpu_out_mat,
		match_num, mat_num_x, mat_num_y);

	cudaDeviceSynchronize();
	*out_mat = new double[mat_num_x * mat_num_y * 9 * (2 * match_num)];


	HANDLE_ERROR(cudaMemcpy(*out_mat, gpu_out_mat, gpu_weighted_A_size, cudaMemcpyDeviceToHost));


	// ///////////////////////////////////////////////////////////////////
	// Release Memory
	// ///////////////////////////////////////////////////////////////////
	HANDLE_ERROR(cudaFree(gpu_weighted_A));
	HANDLE_ERROR(cudaFree(gpu_out_mat));
	gpu_weighted_A = nullptr;
	gpu_out_mat = nullptr;
	if (cpu_weighted_A) {
		delete []cpu_weighted_A;
		cpu_weighted_A = nullptr;
	}
#ifdef DEBUG


	printf("transpose weighted_A = \n");
	for (int j = 0; j < 9; ++j) {
		for (int i = 0; i < 2 * match_num; ++i)
		{
			printf("%f, ", (*out_mat)[j * (2 * match_num) + i]);
		}
		printf("\n");
	}
#endif // DEBUG
	printf("finish transpose\n");

}


void cuda_batch_svd(double * obj_mat, const int &numMatrices, const int &M, const int &N, 
	double **h_V)
{
	printf("%d matrix, (row, col) = (%d, %d)\n", numMatrices, M, N);
	int lda = M;
	

	double *gpu_obj_mat;
	cudaMalloc(&gpu_obj_mat, numMatrices * M * N * sizeof(double));
	cudaMemcpy(gpu_obj_mat, obj_mat, numMatrices * M * N * sizeof(double), cudaMemcpyHostToDevice);

	// --- host side SVD results space
	double *h_S = (double *)malloc(N * numMatrices * sizeof(double));
	*h_V = NULL;
	*h_V = (double *)malloc(numMatrices * M * N * sizeof(double));

	// --- device side SVD workspace and matrices
	int work_size = 0;

	int *devInfo;        cudaMalloc(&devInfo, sizeof(int));
	double *d_S;         cudaMalloc(&d_S, N * numMatrices * sizeof(double));
	double *d_U = NULL;
	double *d_V = NULL;
	HANDLE_ERROR(cudaMalloc(&d_U, M * M * numMatrices * sizeof(double)));
	HANDLE_ERROR(cudaMalloc(&d_S, N * numMatrices * sizeof(double)));
	HANDLE_ERROR(cudaMalloc(&d_V, N * M * numMatrices * sizeof(double)));

	double *d_work = NULL; /* devie workspace for gesvdj */
	int devInfo_h = 0; /* host copy of error devInfo_h */

	// --- Parameters configuration of Jacobi-based SVD
	const double            tol = 1.e-7;
	const int               maxSweeps = 15;
	cusolverEigMode_t jobz;                                   // --- CUSOLVER_EIG_MODE_VECTOR - Compute eigenvectors; CUSOLVER_EIG_MODE_NOVECTOR - Compute singular values only
	jobz = CUSOLVER_EIG_MODE_VECTOR;

	const int               econ = 0;                            // --- econ = 1 for economy size 

	// --- Numerical result parameters of gesvdj 
	double                  residual = 0;
	int                     executedSweeps = 0;

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

	HANDLE_ERROR(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(h_S, d_S, sizeof(double) * N * numMatrices, cudaMemcpyDeviceToHost));
	/*std::cout << "h_S = " << std::endl;
	for (int i = 0; i < N; ++i) {
		std::cout << h_S[i] << ", ";
		std::cout << std::endl;
	}*/
	HANDLE_ERROR(cudaMemcpy(*h_V, d_V, sizeof(double) * N * M * numMatrices, cudaMemcpyDeviceToHost));
	printf("h_V = \n");
	for (int k = 0; k < 1; k++)
	{
		for (int q = 0; q < N; q++) {
			for (int p = 0; p < N; p++) {
				printf("%f, ", (*h_V)[N * M * k + q * M + p]);
			}
			printf("\n");
		}
		printf("---------\n");
	}
	//
	// --- Free resources
	if (gpu_obj_mat) HANDLE_ERROR(cudaFree(gpu_obj_mat));
	if (d_U) HANDLE_ERROR(cudaFree(d_U));
	if (d_S) HANDLE_ERROR(cudaFree(d_S));
	if (d_V) HANDLE_ERROR(cudaFree(d_V));

	if (devInfo) HANDLE_ERROR(cudaFree(devInfo));
	if (d_work) HANDLE_ERROR(cudaFree(d_work));
	if (solver_handle) cusolverDnDestroy(solver_handle);
	if (gesvdj_params) cusolverDnDestroyGesvdjInfo(gesvdj_params);

	HANDLE_ERROR(cudaDeviceReset());
}


void cuda_APAP(const int &src_img_rows, const int &src_img_cols, const int &dst_img_rows, const int &dst_img_cols,
	const int &offset_x, const int &offset_y,
	float (*homo)[3][3], const int &x_num, const int &y_num, const cv::Mat &src_gray, cv::Mat &dst)
{
	float *gpu_src_img = nullptr;
	float *gpu_transform_img = nullptr;
	float *gpu_optical_flow = nullptr;
	float (*gpu_homo)[3][3] = nullptr;

	size_t src_img_size = src_img_cols * src_img_rows * sizeof(float);
	size_t dst_img_size = dst_img_cols * dst_img_rows * sizeof(float);
	size_t homo_size = x_num * y_num * 9 * sizeof(float);

	HANDLE_ERROR(cudaMalloc((void **)& gpu_src_img, src_img_size));
	HANDLE_ERROR(cudaMalloc((void **)& gpu_transform_img, dst_img_size));
	HANDLE_ERROR(cudaMalloc((void **)& gpu_optical_flow, 2 * dst_img_size));
	HANDLE_ERROR(cudaMalloc((void **)& gpu_homo, homo_size));


	// memory copy kernel and src_gray from host to device
	HANDLE_ERROR(cudaMemcpy(gpu_src_img, src_gray.data, src_img_size, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(gpu_homo, homo, homo_size, cudaMemcpyHostToDevice));

	// //////////////////////////////////////////////////////////////////////////////////////////////
	// resident thread; every pixel of result correspond to a thread;
	// //////////////////////////////////////////////////////////////////////////////////////////////
	//int thread_num = CudaGetThreadNum();
	int block_dim_x = threads_x_per_block;
	int block_dim_y = threads_y_per_block;
	int grid_dim_x = (dst_img_cols - 1) / block_dim_x + 1;
	int grid_dim_y = (dst_img_rows - 1) / block_dim_y + 1;
	std::cout << "block size = " << block_dim_x << ", " << block_dim_y << std::endl;
	std::cout << "grid size = " << grid_dim_x << ", " << grid_dim_y << std::endl;

	dim3 grid_size(grid_dim_x, grid_dim_y, 1);
	dim3 block_size(block_dim_x, block_dim_y, 1);
	kernel_APAP <<< grid_size, block_size >> > (gpu_src_img,
		gpu_transform_img, gpu_optical_flow, gpu_homo, y_num, x_num,
		dst_img_rows, dst_img_cols, offset_x, offset_y, src_img_rows, src_img_cols);

	cudaDeviceSynchronize();
	float * transform_result = new float[dst_img_cols * dst_img_rows];
	float * optical_flow = new float[dst_img_cols * dst_img_rows * 2];

	HANDLE_ERROR(cudaMemcpy(transform_result, gpu_transform_img, dst_img_size, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(optical_flow, gpu_optical_flow, dst_img_size * 2, cudaMemcpyDeviceToHost));

	
	dst = cv::Mat(src_img_rows, src_img_cols, CV_32FC1, transform_result).clone();

	// ///////////////////////////////////////////////////////////////////
	// Release Memory
	// ///////////////////////////////////////////////////////////////////
	HANDLE_ERROR(cudaFree(gpu_src_img));
	HANDLE_ERROR(cudaFree(gpu_transform_img));
	HANDLE_ERROR(cudaFree(gpu_optical_flow));
	gpu_src_img = nullptr;
	gpu_transform_img = nullptr;
	gpu_optical_flow = nullptr;


	if (transform_result) { delete[] transform_result; transform_result = nullptr; }
	if (optical_flow) { delete[] optical_flow; optical_flow = nullptr; }

	cudaDeviceReset();

}


void get_least_singular_vector(double *h_V, float(**homo)[3][3], 
	const int &x_num, const int &y_num, const int &match_num)
{
	// h_V: y_num * x_num * 9 * (match_num * 2) 
	*homo = new float[x_num * y_num][3][3];
	for (int i = 0; i < y_num; ++i) 
	{
		for (int j = 0; j < x_num; ++j) 
		{
			int last_singular_index = (i * x_num + j) * (match_num * 2) * 9 +
				(9 - 1) * (match_num * 2);
			int index = 0;
			float factor = (float)h_V[last_singular_index + 8];
			for (int k = 0; k < 3; ++k) {

				for (int l = 0; l < 3; ++l)
				{
					(*homo)[i * x_num + j][k][l] = (float)h_V[last_singular_index + index] / factor;
					index++;
				}
			}
		}
	}
	if (h_V) { delete[] h_V; h_V = nullptr; };
#ifdef  DEBUG
	for (int i = 0; i < 3; ++i) {
		printf("--------------------------------\n");
		for (int k = 0; k < 3; ++k) {

			for (int l = 0; l < 3; ++l)
			{
				printf("%f, ", (*homo)[i][k][l]);
			}
			printf("\n");
		}

	}
#endif //  DEBUG
}



void cuda_main(cv::Mat & src, cv::Mat & dst, cv::Mat &res_img,
	float src_match_pts[][2], float dst_match_pts[][2], int match_num, bool is_stitching = false)
{
	// read src_gray and convert to gray image
	float(*A)[9];
	get_DLT_matrix(src_match_pts, dst_match_pts, match_num, &A);
	int new_dst_cols, new_dst_rows, offset_x, offset_y;
	if (is_stitching) {
		final_size(A, match_num, src.rows, src.cols, dst.rows, dst.cols,
			new_dst_cols, new_dst_rows, offset_x, offset_y);
	}
	else {
		new_dst_cols = dst.cols;
		new_dst_rows = src.cols;
		offset_x = offset_y = 0;
	}
	printf("(w, h, ox, oy) = (%d, %d, %d, %d)\n", new_dst_cols, new_dst_rows, offset_x, offset_y);
	cv::Mat src_gray;
	if (src.type() == CV_8UC3) {
		cv::cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);
		printf("origin gray img\n");
		// uchar to float
		src_gray.convertTo(src_gray, CV_32FC1);
	}
	else {
		src_gray = src.clone();
	}
	cv::normalize(src_gray, src_gray, 1.0, 0, cv::NORM_MINMAX);
	cv::Mat dst_gray;
	if (dst.type() == CV_8UC3) {
		cv::cvtColor(dst, dst_gray, cv::COLOR_BGR2GRAY);
		printf("origin gray img\n");
		// uchar to float
		dst_gray.convertTo(dst_gray, CV_32FC1);
	}
	else {
		dst_gray = dst.clone();
	}

	int dst_img_cols = dst_gray.cols;
	int dst_img_rows = dst_gray.rows;

	// ////////////////////////////////////////////////////////////////////////
	// get weighted A
	// ////////////////////////////////////////////////////////////////////////
	int patch_size_x = threads_x_per_block;   // patch or cell size
	int patch_size_y = threads_y_per_block;
	int x_num;
	int y_num;
	double(*cpu_weighted_A)[9];
	cuda_get_weighted_A(patch_size_x, patch_size_y, 
		new_dst_rows, new_dst_cols, offset_x, offset_y,
		dst_match_pts, match_num, A, 
		x_num, y_num, &cpu_weighted_A);
	std::cout << std::endl << "finish get_weighted_A" << std::endl;


	// ///////////////////////////////////////////////////////////////////////
	// transpose weighted_A
	// ///////////////////////////////////////////////////////////////////////
	match_num = MIN(match_num, 16);
	double *transpose_mat;
	cuda_transpose(cpu_weighted_A, &transpose_mat,
		x_num, y_num, match_num);

	// ///////////////////////////////////////////////////////////////////////
	// batch svd: svd weighted_A, output h_V
	// ///////////////////////////////////////////////////////////////////////
	int num_matrix = x_num * y_num;
	double *h_V;
	cuda_batch_svd(transpose_mat, num_matrix, match_num * 2, 9, &h_V);
	if (transpose_mat) { delete[] transpose_mat; transpose_mat = nullptr; }
	std::cout << std::endl << "finish batch SVD" << std::endl;

	// ///////////////////////////////////////////////////////////////////////
	// use right sigular vector V to do transform
	// ///////////////////////////////////////////////////////////////////////
	int src_img_cols = src_gray.cols;
	int src_img_rows = src_gray.rows;
	float (*homo_lst)[3][3];
	get_least_singular_vector(h_V, &homo_lst,
		x_num, y_num, match_num);
	
	cuda_APAP(src_img_rows, src_img_cols, new_dst_rows, new_dst_cols, offset_x, offset_y, homo_lst, x_num, y_num, src_gray, res_img);
	std::cout << std::endl << "finish cuda_APAP" << std::endl;
	
	cv::normalize(dst_gray, dst_gray, 1.0, 0.0, cv::NORM_MINMAX);
	cv::normalize(res_img, res_img, 1.0, 0.0, cv::NORM_MINMAX);
	dst_gray.convertTo(dst_gray, CV_8UC1, 255.0, 0.0);
	res_img.convertTo(res_img, CV_8UC1, 255.0, 0.0);
	cv::cvtColor(dst_gray, dst_gray, cv::COLOR_GRAY2BGR);
	cv::cvtColor(res_img, res_img, cv::COLOR_GRAY2BGR);
	cv::applyColorMap(res_img, res_img, cv::COLORMAP_JET);
	cv::Mat fusion = 0.5 * res_img + dst_gray * 0.5;
	fusion.convertTo(fusion, CV_8UC3);
	
	cv::imshow("fusion", fusion);
	cv::imshow("res_img", res_img);
	cv::imwrite("res_img.jpg", res_img);
	cv::imwrite("fusion.jpg", fusion);


	//cv::imshow("src", dst_gray);

	cv::waitKey(0);
}




int main()
{
	// dir_1: dst_img_dir, dir_2: dst_img_dir
	// transform src_img to dst_img
	std::string dir_1("../images/ir-ir/2.bmp");
	std::string dir_2("../images/ir-ir/1.bmp");
	//std::string dir_1("../images/others/bridge01.jpg");
	//std::string dir_2("../images/others/bridge02.jpg");

	float(*dst_img_match_pts)[2];
	float(*src_img_match_pts)[2];
	int match_num;
	try {
		match(dir_1, dir_2, &src_img_match_pts, &dst_img_match_pts, match_num);
	}
	catch(int e){
		printf("cannot find enouth match points!");
		return -1;
	}
	/*for (int i = 0; i < match_num; ++i)
	{
		std::cout << src_img_match_pts[i][0] << ", " << src_img_match_pts[i][1] << std::endl;
	}
	for (int i = 0; i < match_num; ++i)
	{
		std::cout << dst_img_match_pts[i][0] << ", " << dst_img_match_pts[i][1] << std::endl;
	}*/
	cv::Mat res_img;
	cv::Mat src_img = cv::imread(dir_1);
	cv::Mat dst_img = cv::imread(dir_2);

	cuda_main(src_img, dst_img, res_img, src_img_match_pts, dst_img_match_pts, match_num);

	if (dst_img_match_pts) { delete[] dst_img_match_pts; dst_img_match_pts = nullptr;}
	if (src_img_match_pts) { delete[] src_img_match_pts; src_img_match_pts = nullptr; }

	
	return 0;
}

