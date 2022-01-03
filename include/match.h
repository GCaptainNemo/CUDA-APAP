#pragma once
#include <vector>

void match(std::string &img1_address, std::string &img2_address, float(**src_match_pts)[2], float(**dst_match_pts)[2], int &match_num, 
	const char *feature_pts_option = "SIFT", const char *match_option = "2nn", const bool &SELF_RANSAC=true);

