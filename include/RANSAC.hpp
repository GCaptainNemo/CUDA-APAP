#pragma once
#include <vector>
#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include <math.h>
#include <stdlib.h>
#include <time.h> 

// #define DEBUG

class RANSAC_obj 
{
public:
	RANSAC_obj(float prob = 0.995, float inlier_p = 0.5, int threshold = 10, int m = 6) :_prob(prob), _inlier_p(inlier_p), _threshold(threshold), _m(m)
	{
		srand((unsigned)time(NULL));  // random seed, seed different -> random val different
	};

	~RANSAC_obj() {};

	int get_N();

	// sample _m points
	void sample_index_lst(const int &min_index, const int &max_index, const int &m, int * const index_lst);

	// calculate homography
	template<class T>
	void calculate_homography(const std::vector<cv::Point2f> &src, const std::vector<cv::Point2f> &dst, 
		const int * const index_lst, T homography[3][3]);

	// get project distance
	template<class T>
	T get_distance(const cv::Point2f &src_pts, const cv::Point2f &dst_pts, T homography[3][3]);

	void start_RANSAC(const std::vector<cv::Point2f> &src, const std::vector<cv::Point2f> &dst, 
		std::vector<uchar> &ransac_status, int max_sampling=1000);

private:
	float _prob, _inlier_p;
	int _threshold, _m;

};



void RANSAC_obj::sample_index_lst(const int &min_index, const int &max_index, const int &m, int * const index_lst)
{
	// sampling m integer number between [min_index, max_index]
	for (int i = 0; i < m; ++i) {
		int index = rand() % (max_index - min_index + 1) + min_index;
		index_lst[i] = index;
	}
#ifdef DEBUG
	printf("index_lst = \n");
	for (int i = 0; i < m; ++i) {
		printf("%d, ", index_lst[i]);
	}

#endif
}

int RANSAC_obj::get_N() {
	// calculate sampling times

	int sample_times = static_cast<int>(log(1 - this->_prob) / log(1 - pow(this->_inlier_p, this->_m) + 1e-8));
	return sample_times;

}

template<class T>
void RANSAC_obj::calculate_homography(const std::vector<cv::Point2f> &src, const std::vector<cv::Point2f> &dst, 
	const int * const index_lst, T homography[3][3])
{
	// calculate src -to - dst homography transform
	cv::Mat A(_m * 2, 9, CV_32FC1);
	for (int row = 0; row < _m; ++row)
	{
		int pts_idx = index_lst[row];
		float x1 = src[pts_idx].x;
		float x2 = src[pts_idx].y;
		float y1 = dst[pts_idx].x;
		float y2 = dst[pts_idx].y;
		A.at<float>(row * 2, 0) = x1;
		A.at<float>(row * 2, 1) = x2;
		A.at<float>(row * 2, 2) = 1.0f;
		A.at<float>(row * 2, 3) = 0.0f;
		A.at<float>(row * 2, 4) = 0.0f;
		A.at<float>(row * 2, 5) = 0.0f;
		A.at<float>(row * 2, 6) = -x1 * y1;
		A.at<float>(row * 2, 7) = -x2 * y1;
		A.at<float>(row * 2, 8) = -1. * y1;

		// 
		A.at<float>(row * 2 + 1, 0) = 0.0f;
		A.at<float>(row * 2 + 1, 1) = 0.0f;
		A.at<float>(row * 2 + 1, 2) = 0.0f;
		A.at<float>(row * 2 + 1, 3) = x1;
		A.at<float>(row * 2 + 1, 4) = x2;
		A.at<float>(row * 2 + 1, 5) = 1.0f;
		A.at<float>(row * 2 + 1, 6) = -x1 * y2;
		A.at<float>(row * 2 + 1, 7) = -x2 * y2;
		A.at<float>(row * 2 + 1, 8) = -1. * y2;
	}
	cv::Mat u, w, vt;
	cv::SVDecomp(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
	int row_index = vt.rows - 1;
	int index = 0;
	T factor = (T)vt.at<float>(row_index, 8);
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			homography[i][j] = (T)vt.at<float>(row_index, index) / factor;
			index++;
		}
	}
	/*#ifdef DEBUG
	printf("homography = \n");
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			printf("%f, ", homography[i][j]);
		}
		printf("\n");
	}

	#endif*/
};


template<class T>
T RANSAC_obj::get_distance(const cv::Point2f &src_pts, const cv::Point2f &dst_pts, T homography[3][3]) 
{

	T x1 = (T)src_pts.x;
	T x2 = (T)src_pts.y;
	T y1 = (T)dst_pts.x;
	T y2 = (T)dst_pts.y;
	T denominator = x1 * homography[2][0] + x2 * homography[2][1] + 1.0 * homography[2][2];
	T trans_x1 = (x1 * homography[0][0] + x2 * homography[0][1] + 1.0 * homography[0][2]) / denominator;
	T trans_x2 = (x1 * homography[1][0] + x2 * homography[1][1] + 1.0 * homography[1][2]) / denominator;
	T dist = sqrt(pow(y1 - trans_x1, 2) + pow(y2 - trans_x2, 2));
//#ifdef DEBUG
//	printf("homography = \n");
//	for (int i = 0; i < 3; ++i) {
//		for (int j = 0; j < 3; ++j) {
//			printf("%f, ", homography[i][j]);
//		}
//		printf("\n");
//	}
//	printf("dist = %f\n", dist);
//#endif
	return dist;
};


void RANSAC_obj::start_RANSAC(const std::vector<cv::Point2f> &src, const std::vector<cv::Point2f> &dst, 
	std::vector<uchar> &ransac_status, int max_sampling)
{
	int num = this->get_N();
	int sampling_num = MIN(num, max_sampling);
	int *index_lst = new int[src.size()];
	int match_size = src.size();
	
	printf("sampling %d times\n", sampling_num);
	printf("matchsize = %d\n", match_size);

	int max_inlier_num = -1;
	for (int i = 0; i < sampling_num; i++)
	{
		sample_index_lst(0, src.size() - 1, _m, index_lst);
		double g_homography[3][3];
		this->calculate_homography<double>(src, dst, index_lst, g_homography);
		std::vector<uchar> linshi_ransac_status(match_size);

		// calculate inlier points
		int inlier_num = 0;
		for (int j = 0; j < match_size; ++j) {
			double dist = get_distance<double>(src[j], dst[j], g_homography);
			if (dist < _threshold) {
				linshi_ransac_status[j] = 1;
				inlier_num += 1;
			}
			else {
				linshi_ransac_status[j] = 0;
			}
		}
		if (inlier_num > max_inlier_num) {
			max_inlier_num = inlier_num;
			ransac_status.swap(linshi_ransac_status);
		}
	}
	printf("there are %d inlier points\n", max_inlier_num);


	if (index_lst) { delete[]index_lst; index_lst = nullptr; }

};




