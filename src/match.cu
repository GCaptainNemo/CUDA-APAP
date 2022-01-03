#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include "../include/RANSAC.hpp"



using namespace std;


void match(std::string &img1_address, std::string &img2_address, 
	float (**src_match_pts)[2], float(**dst_match_pts)[2], int &match_num, const char *feature_pts_option, const char *match_option, const bool &SELF_RANSAC)
{
	cv::Mat src = cv::imread(img1_address);
	cv::Mat dst = cv::imread(img2_address);

	// define detector, descriptor and matcher
	cv::Ptr<cv::FeatureDetector> detector;
	cv::Ptr<cv::DescriptorExtractor> descriptor;
	cv::Ptr<cv::DescriptorMatcher> matcher;
	if (strcmp(feature_pts_option, "ORB") == 0) {
		detector = cv::ORB::create();
		descriptor = cv::ORB::create();
		matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
	}
	else if (strcmp(feature_pts_option, "SIFT") == 0) {
		detector = cv::SIFT::create();
		descriptor = cv::SIFT::create();
		matcher = cv::DescriptorMatcher::create("FlannBased");
	}


	std::vector<cv::KeyPoint>keyPoints1;
	std::vector<cv::KeyPoint>keyPoints2;


	// 1. detect key points
	detector->detect(src, keyPoints1);
	detector->detect(dst, keyPoints2);

	// 2. calculate descriptor according to corner points
	
	cv::Mat description1;
	cv::Mat description2;
	descriptor->compute(src, keyPoints1, description1);
	descriptor->compute(dst, keyPoints2, description2);

	std::cout << "key pts size = " << keyPoints1.size() << std::endl;
	std::cout << "description1 col = "<< description1.cols << std::endl;      //列数,描述子维度
	std::cout << "description1 row = " << description1.rows << std::endl;      //行数，特征点个数


	// 3. Brute Force Match (2nn is more robust than 1nn filtering methods)
	std::vector<cv::DMatch> good_matches;

	if (strcmp(match_option, "1nn") == 0) {
		std::vector <cv::DMatch> coarse_matches;
		matcher->match(description1, description2, coarse_matches);
		
		// 4. filter 1nn match pts according to descriptor distance
		auto min_max = minmax_element(coarse_matches.begin(), coarse_matches.end(),
			[](const cv::DMatch & m1, const cv::DMatch & m2) {return m1.distance < m2.distance; });
		double min_dist = min_max.first->distance;
		double max_dist = min_max.second->distance;
		std::cout << "min dist = " << min_dist << std::endl;
		std::cout << "max dist = " << max_dist << std::endl;
		for (int i = 0; i < description1.rows; ++i)
		{
			if (coarse_matches[i].distance <= (2 * min_dist > 30.0 ? 2 * min_dist : 30.0))
			{
				good_matches.push_back(coarse_matches[i]);
			}
		}
	}
	else if(strcmp(match_option, "2nn") == 0){
		std::vector <std::vector<cv::DMatch>> coarse_matches;
		matcher->knnMatch(description1, description2, coarse_matches, 2);
		// 4. filter 1nn match pts according to relative descriptor distance
		float min_ratio = 0.7;
		for (int i = 0; i < coarse_matches.size(); i++) {
			const cv::DMatch bestMatch = coarse_matches[i][0];
			const cv::DMatch betterMatch = coarse_matches[i][1];
			if (bestMatch.distance < betterMatch.distance * min_ratio)
				good_matches.push_back(bestMatch);
		}
	}
	//// draw match	
	//cv::Mat img_filter;
	//cv::drawMatches(src, keyPoints1, dst, keyPoints2, good_matches, img_filter, cv::Scalar(0, 255, 0), cv::Scalar::all(-1));
	//cv::imshow("AFTER filter", img_filter);

	std::vector<cv::KeyPoint> R_keypoint01, R_keypoint02;
	for (int i = 0; i < good_matches.size(); i++)
	{
		R_keypoint01.push_back(keyPoints1[good_matches[i].queryIdx]);
		R_keypoint02.push_back(keyPoints2[good_matches[i].trainIdx]);
		
	}

	// keypts -> point2f
	std::vector<cv::Point2f>p01, p02;
	for (int i = 0; i < good_matches.size(); i++)
	{
		p01.push_back(R_keypoint01[i].pt);
		p02.push_back(R_keypoint02[i].pt);
	}
	printf("before ransac there are %d points\n ", p01.size());
	
	
	// 5. filter using RANSAC_obj algorithm.
	// homography ransac is more robust than fundamental ransac
	std::vector<uchar> RansacStatus;
	if (SELF_RANSAC) {
		RANSAC_obj homo_ransac_obj;
		homo_ransac_obj.start_RANSAC(p01, p02, RansacStatus);
	}
	else {
		cv::Mat Fundamental = cv::findHomography(p01, p02, RansacStatus, cv::RANSAC);
		//float ransacReprojThreshold = 0.1;   
		//cv::Mat Fundamental = cv::findFundamentalMat(p01, p02, RansacStatus, cv::RANSAC, ransacReprojThreshold);
	}
	std::vector<cv::KeyPoint> RR_keypoint01, RR_keypoint02;
	std::vector<cv::DMatch> RR_matches;            //重新定义RR_keypoint 和RR_matches来存储新的关键点和匹配矩阵
	int index = 0;
	for (int i = 0; i < good_matches.size(); i++)
	{
		if (RansacStatus[i] != 0)
		{
			RR_keypoint01.push_back(R_keypoint01[i]);
			RR_keypoint02.push_back(R_keypoint02[i]);
			good_matches[i].queryIdx = index;
			good_matches[i].trainIdx = index;
			RR_matches.push_back(good_matches[i]);
			index++;
		}
	}
	std::cout << "matched points num:" << RR_matches.size() << std::endl;
	if (index < 4) { throw - 1; };
	// ///////////////////////////////////////////
	// draw result
	// ///////////////////////////////////////////
	cv::Mat img_RR_matches;
	cv::drawMatches(src, RR_keypoint01, dst, RR_keypoint02, RR_matches, img_RR_matches, cv::Scalar(0, 255, 0), cv::Scalar::all(-1));
	cv::namedWindow("AFTER RANSAC", cv::WINDOW_NORMAL);
	cv::imshow("AFTER RANSAC", img_RR_matches);
	cv::imwrite("MATCH.jpg", img_RR_matches);

	cv::waitKey(0);

	// ///////////////////////////////////////////
	// output
	// ///////////////////////////////////////////
	match_num = RR_matches.size();
	*src_match_pts = new float[match_num][2];
	*dst_match_pts = new float[match_num][2];
	for (int i = 0; i < match_num; ++i)
	{
		(*src_match_pts)[i][0] = RR_keypoint01[i].pt.x;
		(*src_match_pts)[i][1] = RR_keypoint01[i].pt.y;
		(*dst_match_pts)[i][0] = RR_keypoint02[i].pt.x;
		(*dst_match_pts)[i][1] = RR_keypoint02[i].pt.y;
		printf("(%f, %f)-(%f, %f)\n", RR_keypoint01[i].pt.x, RR_keypoint01[i].pt.y,
			RR_keypoint02[i].pt.x, RR_keypoint02[i].pt.y);
	}
	std::cout << "output match pts finish !!" << std::endl;

};