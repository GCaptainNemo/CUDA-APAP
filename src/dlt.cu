#include "dlt.h"
#include <algorithm>

// input src_img match points and dst_img match points
// output DLT matrix A(dst img to src img) 
// minimize Ah s.t, ||h|| = 1
void get_DLT_matrix(const float src_img_match_pts[][2], const float dst_img_match_pts[][2], int match_size, float(**A)[9])
{
	int size = match_size;
	*A = new float[size * 2][9];
	for (int row = 0; row < size; ++row)
	{
		float x1 = dst_img_match_pts[row][0];
		float x2 = dst_img_match_pts[row][1];
		float y1 = src_img_match_pts[row][0];
		float y2 = src_img_match_pts[row][1];
		(*A)[row * 2][0] = x1;
		(*A)[row * 2][1] = x2;
		(*A)[row * 2][2] = 1.0f;
		(*A)[row * 2][3] = 0.0f;
		(*A)[row * 2][4] = 0.0f;
		(*A)[row * 2][5] = 0.0f;
		(*A)[row * 2][6] = -x1 * y1;
		(*A)[row * 2][7] = -x2 * y1;
		(*A)[row * 2][8] = -1. * y1;

		// 
		(*A)[row * 2 + 1][0] = 0.0f;
		(*A)[row * 2 + 1][1] = 0.0f;
		(*A)[row * 2 + 1][2] = 0.0f;
		(*A)[row * 2 + 1][3] = x1;
		(*A)[row * 2 + 1][4] = x2;
		(*A)[row * 2 + 1][5] = 1.0f;
		(*A)[row * 2 + 1][6] = -x1 * y2;
		(*A)[row * 2 + 1][7] = -x2 * y2;
		(*A)[row * 2 + 1][8] = -1. * y2;
	}
}

void final_size(float(*A)[9], const int &match_num, const int &src_img_row, const int &src_img_col,
	const int &dst_img_row, const int &dst_img_col, 
	int &new_dst_width, int &new_dst_height, int &offset_x, int &offset_y)
{
	cv::Mat A_mat(match_num * 2, 9, CV_32FC1, A);

	cv::Mat u, w, vt;
	cv::SVDecomp(A_mat, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
	int row_index = vt.rows - 1;
	int index = 0;
	cv::Mat homography(3, 3, CV_32FC1);
	float factor = vt.at<float>(row_index, 8);
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			homography.at<float>(i, j) = vt.at<float>(row_index, index) / factor;
			index++;
		}
	}
	cv::Mat src_to_dst_homography;
	cv::invert(homography, src_to_dst_homography);

	// ////////////////////////////////////////
	float width = (float)src_img_col;
	float height = (float)src_img_row;
	float leftup_[3] = { 0.0, 0.0, 1.0 };
	float rightup_[3] = { width, 0.0, 1.0 };
	float leftdown_[3] = { 0.0, height, 1.0 };
	float rightdown_[3] = { width, height, 1.0 };

	cv::Mat leftup(3, 1, CV_32FC1, leftup_);
	cv::Mat rightup(3, 1, CV_32FC1, rightup_);
	cv::Mat leftdown(3, 1, CV_32FC1, leftdown_);
	cv::Mat rightdown(3, 1, CV_32FC1, rightdown_);
	std::vector<float> x_lst;
	std::vector<float> y_lst;

	cv::Mat linshi_ = src_to_dst_homography * leftup;
	x_lst.push_back(linshi_.at<float>(0, 0) / linshi_.at<float>(2, 0));
	y_lst.push_back(linshi_.at<float>(1, 0) / linshi_.at<float>(2, 0));
	linshi_ = src_to_dst_homography * rightup;
	x_lst.push_back(linshi_.at<float>(0, 0) / linshi_.at<float>(2, 0));
	y_lst.push_back(linshi_.at<float>(1, 0) / linshi_.at<float>(2, 0));
	linshi_ = src_to_dst_homography * leftdown;
	x_lst.push_back(linshi_.at<float>(0, 0) / linshi_.at<float>(2, 0));
	y_lst.push_back(linshi_.at<float>(1, 0) / linshi_.at<float>(2, 0));
	linshi_ = src_to_dst_homography * rightdown;
	x_lst.push_back(linshi_.at<float>(0, 0) / linshi_.at<float>(2, 0));
	y_lst.push_back(linshi_.at<float>(1, 0) / linshi_.at<float>(2, 0));

	float max_x = MAX(*std::max_element(x_lst.begin(), x_lst.end()), dst_img_col);
	float min_x = MIN(*std::min_element(x_lst.begin(), x_lst.end()), 0.0);
	float max_y = MAX(*std::max_element(y_lst.begin(), y_lst.end()), dst_img_row);
	float min_y = MIN(*std::min_element(y_lst.begin(), y_lst.end()), 0.0);
	new_dst_width = max_x - min_x;
	new_dst_height = max_y - min_y;
	offset_x = (min_x < 0 ? -min_x : 0);
	offset_y = (min_y < 0 ? -min_y : 0);




};

