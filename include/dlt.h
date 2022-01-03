#pragma once

#include <vector>
#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>

void get_DLT_matrix(const float src_img_match_pts[][2], const float dst_img_match_pts[][2], int match_size, float(**A)[9]);

void final_size(float(*A)[9], const int &match_num, const int &src_img_row, const int &src_img_col, 
	const int &dst_img_row, const int &dst_img_col, 
	int &new_dst_width, int &new_dst_height, int &offset_x, int &offset_y);

