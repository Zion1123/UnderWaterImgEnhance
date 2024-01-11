#include"UnderWaterImg.h"
#include<ctime>
using namespace cv;
using namespace std;


// 用于对比度增强
// 幂次变换
cv::Mat powertransform(cv::Mat &input, float index)
{
	cv::Mat result = input.clone();
	float c = 255 / pow(255, index);//系数(满足映射关系)
	for (int i = 0; i < input.rows; i++)
	{
		for (int j = 0; j < input.cols; j++)
		{
			result.at<uchar>(i, j) = cv::saturate_cast<uchar>(c * pow(input.at<uchar>(i, j), index));
		}
	}
	return result;
}


//直方图拉伸
cv::Mat histStretch(cv::Mat src)
{
	int min = 256, max = -1;
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			int I = src.at<uchar>(i, j);
			if (I>max)
			{
				max = I;
			}
			if (I < min)
			{
				min = I;
			}
		}
	}

	cv::Mat result = src.clone();
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			int a = round((255 / (max - min) + 0.5)*(src.at<uchar>(i, j) - min));
			if (a < 0) a = 0;
			if (a > 255) a = 255;
			result.at<uchar>(i, j) = a;
		}
	}
	return result;
}

//图像锐化
cv::Mat Sharpen_Img(cv::Mat src)
{
	cv::Mat Guass_Img;
	GaussianBlur(src, Guass_Img, cv::Size(5, 5), 0, 0);

	cv::Mat sub_img = src - Guass_Img;
	//imwrite("减法.bmp", sub_img);


	cv::Mat N = histStretch(sub_img);
	//cv::imwrite("直方图拉伸.bmp", N);

	cv::Mat result = src + N;
	return result;
}



//暗通道
cv::Mat getDarkChannelImg(const Mat src, const int r)
{
	int height = src.rows;
	int width = src.cols;
	Mat darkChannelImg(src.size(), CV_8UC1);

	//把图像分成patch,求patch框内的最小值,得到dark_channel image
	//r is the patch radius, patchSize=2*r+1 
	//这一步实际上是最小值滤波的过程
	cv::Mat rectImg;
	int patchSize = 2 * r + 1;
	for (int j = 0; j<height; j++)
	{
		for (int i = 0; i<width; i++)
		{
			cv::getRectSubPix(src, cv::Size(patchSize, patchSize), cv::Point(i, j), rectImg); //Point横向x 纵向y
			double minValue = 0;
			cv::minMaxLoc(rectImg, &minValue, 0, 0, 0); //get min pix value
			darkChannelImg.at<uchar>(j, i) = cv::saturate_cast<uchar>(minValue);//using saturate_cast to set pixel value to [0,255]  
		}
	}
	return darkChannelImg;
}
//导向滤波
cv::Mat GuidedFilter(cv::Mat I, cv::Mat p, int r, double eps)//r的取值不小于进行最小值滤波的半径的4倍
{
	/*https://zhuanlan.zhihu.com/p/98368439
	% GUIDEDFILTER   O(N) time implementation of guided filter.
	%
	%   - guidance image: I (should be a gray-scale/single channel image)
	%   - filtering input image: p (should be a gray-scale/single channel image)
	%   - local window radius: r
	%   - regularization parameter: eps
	*/

	cv::Mat _I;
	I.convertTo(_I, CV_64FC1, 1.0 / 255);
	I = _I;

	cv::Mat _p;
	p.convertTo(_p, CV_64FC1, 1.0 / 255);
	p = _p;

	//[hei, wid] = size(I);  
	int hei = I.rows;
	int wid = I.cols;

	r = 2 * r + 1;//因为opencv自带的boxFilter（）中的Size,比如9x9,我们说半径为4 

	//mean_I = boxfilter(I, r) ./ N;  
	cv::Mat mean_I;
	cv::boxFilter(I, mean_I, CV_64FC1, cv::Size(r, r));

	//mean_p = boxfilter(p, r) ./ N;  
	cv::Mat mean_p;
	cv::boxFilter(p, mean_p, CV_64FC1, cv::Size(r, r));

	//mean_Ip = boxfilter(I.*p, r) ./ N;  
	cv::Mat mean_Ip;
	cv::boxFilter(I.mul(p), mean_Ip, CV_64FC1, cv::Size(r, r));

	//cov_Ip = mean_Ip - mean_I .* mean_p; % this is the covariance of (I, p) in each local patch.  
	cv::Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);

	//mean_II = boxfilter(I.*I, r) ./ N;  
	cv::Mat mean_II;
	cv::boxFilter(I.mul(I), mean_II, CV_64FC1, cv::Size(r, r));

	//var_I = mean_II - mean_I .* mean_I;  
	cv::Mat var_I = mean_II - mean_I.mul(mean_I);

	//a = cov_Ip ./ (var_I + eps); % Eqn. (5) in the paper;     
	cv::Mat a = cov_Ip / (var_I + eps);

	//b = mean_p - a .* mean_I; % Eqn. (6) in the paper;  
	cv::Mat b = mean_p - a.mul(mean_I);

	//mean_a = boxfilter(a, r) ./ N;  
	cv::Mat mean_a;
	cv::boxFilter(a, mean_a, CV_64FC1, cv::Size(r, r));

	//mean_b = boxfilter(b, r) ./ N;  
	cv::Mat mean_b;
	cv::boxFilter(b, mean_b, CV_64FC1, cv::Size(r, r));

	//q = mean_a .* I + mean_b; % Eqn. (8) in the paper;  
	cv::Mat q = (mean_a.mul(I) + mean_b) * 255;

	return q;
}

//全局大气光强 最小值滤波后取前0.1%像素均值
double getGlobelAtmosphericLight(const Mat src)
{
	int num = src.cols * src.rows * 0.001;
	double A = -1;
	vector<int> pixels;
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			int temp = src.at<uchar>(i, j);
			pixels.push_back(temp);
		}
	}
	sort(pixels.begin(), pixels.end());
	reverse(pixels.begin(), pixels.end());
	double avg = 0;
	double sum = 0;
	for (int i = 0; i <num; i++)
	{
		sum += pixels[i];
	}
	A = sum / num;
	//A = A > 220 ? 200 : A;
	return A    /*> 220 ? 220 : A*/;
}

//投射率图
cv::Mat getTransimissionImg(const double A, const Mat src, const double w)
{
	cv::Mat transmissionImg(src.size(), CV_64FC1);
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			transmissionImg.at<double>(i, j) = 1 - w * (double)src.at<uchar>(i, j) / A;
		}
	}

	return transmissionImg;
}

//本文投射率图计算方法， src为最小值滤波后的图像 
cv::Mat MygetTransimissionImg(const double A, const Mat input, const Mat src, const double w)
{

	//GaussianBlur(input, input, cv::Size(3, 3), 3, 3);
	Mat disimg(src.size(), CV_8UC1);
	Mat everyGauss(input.size(), CV_8UC1);
	GaussianBlur(input, everyGauss, cv::Size(3, 3), 3, 3);

	int height = src.rows;
	int width = src.cols;
	for (int i = 0; i < height; i++){
		for (int j = 0; j < width; j++){
			disimg.at<uchar>(i, j) = abs(input.at<uchar>(i, j) - everyGauss.at<uchar>(i, j));
		}
	}

	for (int k = 2; k <= 4; k++){
		int r = pow(2, k) + 1;

		GaussianBlur(everyGauss, everyGauss, cv::Size(r, r), r, r);

		for (int i = 0; i < height; i++){
			for (int j = 0; j < width; j++){
				disimg.at<uchar>(i, j) += abs(input.at<uchar>(i, j) - everyGauss.at<uchar>(i, j));
			}
		}

	}

	for (int i = 0; i < height; i++){
		for (int j = 0; j < width; j++){
			disimg.at<uchar>(i, j) = disimg.at<uchar>(i, j) / 4;
		}
	}

	//GaussianBlur(disimg, input, cv::Size(3, 3), 3, 3);

	//模糊图
	cv::Mat burimg(src.size(), CV_8UC1);
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			burimg.at<uchar>(i, j) = disimg.at<uchar>(i, j) + src.at<uchar>(i, j);
		}
	}
	GaussianBlur(burimg, burimg, cv::Size(7, 7), 3, 3);

	cv::Mat transmissionImg(src.size(), CV_64FC1);
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			transmissionImg.at<double>(i, j) = 1 - w * (double)burimg.at<uchar>(i, j) / A;
		}
	}

	return transmissionImg;
}

//去雾图像（导向滤波）
cv::Mat getDehazedImg_guidedFilter(Mat src, Mat transmissionImage, double A, int r, double eps)
{
	double tmin = 0.1;
	double tmax = 0;

	Mat dehazedImg = Mat::zeros(src.size(), CV_8UC1);

	//GaussianBlur(transmissionImage, transmissionImage, cv::Size(11,11),5,5);

	Mat new_transmissionImage = GuidedFilter(src, transmissionImage, r, eps);
	for (int i = 0; i<src.rows; i++)
	{
		for (int j = 0; j<src.cols; j++)
		{
			double transmission = new_transmissionImage.at<double>(i, j);
			int srcData = src.at<uchar>(i, j);

			tmax = max(transmission, tmin);
			//(I-A)/t +A  
			dehazedImg.at<uchar>(i, j) = cv::saturate_cast<uchar>(abs((srcData - A) / tmax + A));
		}
	}
	return dehazedImg;

}


// 去雾算法
cv::Mat ImageProcess::DehazedImg(Mat input){
	Mat darkimg = getDarkChannelImg(input, 2);//卷积核大小影响非常明显（过大会区域过曝）
	double A = getGlobelAtmosphericLight(darkimg);
	Mat t1 = getTransimissionImg(A, darkimg, 0.9);
	Mat t = MygetTransimissionImg(A, input, darkimg, 0.9);

	Mat result = getDehazedImg_guidedFilter(input, t1, A, 7, 0.0001);
	Mat my_result = getDehazedImg_guidedFilter(input, t, A, 7, 0.0001);
	//my_result = powertransform(my_result, 0.79);//幂次变换
	return my_result;
}




cv::Mat patch_img(cv::Mat src, int beginx, int beginy, int endx, int endy)
{
	cv::Mat result = cv::Mat(endy - beginy, endx - beginx, CV_64F);
	for (int i = 0; i < result.rows; i++)
	{
		for (int j = 0; j < result.cols; j++)
		{
			result.at<double>(i, j) = src.at<uchar>(beginy + i, beginx + j) / (double)255;
		}
	}
	return result;
}

//图像块分解
cv::Mat path_deal(cv::Mat src, double &L, double &C)
{
	cv::Mat result = cv::Mat(src.rows, src.cols, CV_64F);
	double sum = 0;
	vector<double> pixels;
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			double pixel = src.at<double>(i, j);
			sum += pixel;
			pixels.push_back(pixel);
		}
	}
	//均值
	L = (double)sum / (src.rows * src.cols);

	//查看减去均值后结果
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			src.at<double>(i, j) -= L;
		}
	}

	double sum_pow2 = 0;
	for (int i = 0; i < pixels.size(); i++)
	{
		sum_pow2 += (pixels[i] - L) * (pixels[i] - L);
	}
	//模
	C = sqrt(sum_pow2);

	for (int i = 0; i < result.rows; i++)
	{
		for (int j = 0; j < result.cols; j++)
		{
			result.at<double>(i, j) = (src.at<double>(i, j) - L) / C;
		}
	}

	return result;

}

//高斯金字塔下一层
cv::Mat DOG(cv::Mat src)
{
	cv::Mat Guass_Img;
	GaussianBlur(src, Guass_Img, cv::Size(3, 3), 0, 0);
	cv::Mat result;
	pyrDown(Guass_Img, result, cv::Size(src.cols / 2, src.rows / 2));
	return result;
}

//拉普拉斯第n层(输入: 第n层高斯图像)
cv::Mat LP(cv::Mat src)
{
	cv::Mat nGuass_Img;
	GaussianBlur(src, nGuass_Img, cv::Size(5, 5), 0, 0);

	cv::Mat DownGuass_Img;
	pyrDown(nGuass_Img, DownGuass_Img, cv::Size(src.cols / 2, src.rows / 2));


	cv::Mat up_Img;
	resize(DownGuass_Img, up_Img, cv::Size(src.cols, src.rows), 0, 0, INTER_CUBIC);
	//pyrUp(DownGuass_Img, up_Img, cv::Size(h, w));只能是原shu'rsize的两倍

	cv::Mat Guass_up;
	GaussianBlur(up_Img, Guass_up, cv::Size(5, 5), 0, 0);

	cv::Mat result = nGuass_Img.clone();
	result = src - Guass_up;
	return result;
}

cv::Mat StructFusion(cv::Mat Sc, cv::Mat Ss, double Cc, double Cs)
{
	cv::Mat result = cv::Mat(Sc.rows, Sc.cols, CV_64F);
	cv::Mat Cctr(Sc.rows, Sc.cols, CV_64F);
	cv::Mat Csharp(Sc.rows, Sc.cols, CV_64F);

	for (int i = 0; i < Sc.rows; i++)
	{
		for (int j = 0; j < Sc.cols; j++)
		{
			Cctr.at<double>(i, j) = Cc / (Cc + Cs);
		}
	}

	for (int i = 0; i < Ss.rows; i++)
	{
		for (int j = 0; j < Ss.cols; j++)
		{
			Csharp.at<double>(i, j) = Cs / (Cc + Cs);
		}
	}

	//////融合方法
	////////第一层
	//cv::Mat Gc1 = Cctr; cv::Mat Lc1 = LP(Sc); cv::Mat Gs1 = Csharp; cv::Mat Ls1 = LP(Ss);
	//cv::Mat Gc1 = Cctr; cv::Mat Lc1 = Sc; cv::Mat Gs1 = Csharp; cv::Mat Ls1 = Ss;

	//cv::Mat ctr1, sharp1;
	//multiply(Gc1, Lc1, ctr1); multiply(Gs1, Ls1, sharp1);
	//cv::Mat S1 = cv::Mat(Sc.rows, Sc.cols, CV_64F);
	//S1 = ctr1 + sharp1;

	////第二层
	//cv::Mat Gc2 = DOG(Gc1); cv::Mat Lc2 = LP(DOG(Sc));  cv::Mat Gs2 = DOG(Gs1); cv::Mat Ls2 = (DOG(Ss));

	//cv::Mat ctr2, sharp2;
	//multiply(Gc2, Lc2, ctr2); multiply(Gs2, Ls2, sharp2);
	//cv::Mat S2 = cv::Mat(Sc.rows/2, Sc.cols/2, CV_64F);
	//S2 = ctr2 + sharp2;

	//cv::Mat upS2;
	////pyrUp(S2, upS2, cv::Size(Sc.rows, Sc.cols));
	//resize(S2, upS2, cv::Size(Sc.cols, Sc.rows), 0, 0, CV_INTER_CUBIC);

	//result = S1 + upS2;


	////直接加权法
	cv::Mat Gc1 = Cctr; cv::Mat Lc1 = Sc; cv::Mat Gs1 = Csharp; cv::Mat Ls1 = Ss;

	cv::Mat ctr1, sharp1;
	multiply(Gc1, Lc1, ctr1); multiply(Gs1, Ls1, sharp1);
	cv::Mat S1 = cv::Mat(Sc.rows, Sc.cols, CV_64F);
	S1 = ctr1 + sharp1;
	result = S1;
	return result;

}

//图像块分解融合
cv::Mat jubuSPDF(cv::Mat sharpen_img, cv::Mat contrast_img)
{
	//划分区块 
	int k = 40;

	int h = sharpen_img.rows; int w = sharpen_img.cols;

	//每块的宽高（若不能整除则该方向有(k +1)块）
	int kh = h / k; int kw = w / k;

	cv::Mat result = cv::Mat(sharpen_img.rows, sharpen_img.cols, CV_8U);
	for (int i = 0; i < k; i++)
	{
		for (int j = 0; j < k; j++)
		{
			int beginx = i * kw, beginy = j*kh;
			int endx = (i + 1) * kw, endy = (j + 1) * kh;
			if (w%k != 0 && i == k - 1) endx = w;//不能整除时最后一块区域与上一块合并
			if (h%k != 0 && j == k - 1) endy = h;
			//划块
			cv::Mat cpatch = patch_img(contrast_img, beginx, beginy, endx, endy);
			cv::Mat spatch = patch_img(sharpen_img, beginx, beginy, endx, endy);

			//块分解
			double Lc, Ls, Cc, Cs;
			cv::Mat Sc = path_deal(cpatch, Lc, Cc);
			cv::Mat Ss = path_deal(spatch, Ls, Cs);

			//L的确定
			double sigma = 0.15;
			double Wc = exp(-Lc*Lc*0.5 / (sigma*sigma)) * Lc / (sigma*sigma);
			double Ws = exp(-Ls*Ls*0.5 / (sigma*sigma)) * Ls / (sigma*sigma);
			//double  Wc = Lc;
			//double  Ws = Ls;
			double L = (Wc * Lc) / (Wc + Ws) + (Ws * Ls) / (Wc + Ws);
			double C = max(Cc, Cs);

			cv::Mat Fusion = StructFusion(Sc, Ss, Cc, Cs);

			for (int hh = 0; hh < Fusion.rows; hh++)
			{
				for (int ww = 0; ww < Fusion.cols; ww++)
				{
					result.at<uchar>(beginy + hh, beginx + ww) = cv::saturate_cast<uchar>(((Fusion.at<double>(hh, ww) * C + L) * 255));
				}
			}

			//cv::Mat chakan = cv::Mat(Fusion.rows, Fusion.cols, CV_8U);
			//for (int ww = 0; ww < Fusion.rows; ww++)
			//{
			//	for (int hh = 0; hh < Fusion.cols; hh++)
			//	{
			//		chakan.at<uchar>(ww,hh) = cv::saturate_cast<uchar>(((Fusion.at<double>(ww, hh) * C + L) * 255));
			//	}
			//}
			//imwrite("查看.bmp", chakan);

		}
	}
	return result;
}

cv::Mat ImageProcess::SPDF(cv::Mat src){
	cv::Mat sharpen = Sharpen_Img(src);
	sharpen = powertransform(src,0.65);

	cv::Mat contrast = powertransform(src, 0.5);

	// 融合锐化图像和对比度图像
	cv::Mat juburesult = jubuSPDF(sharpen, contrast);
	medianBlur(juburesult, juburesult, 3);
	GaussianBlur(juburesult, juburesult, cv::Size(3, 3), 0, 0);
	//return juburesult;

	cv::Mat gamma = powertransform(juburesult, 0.6);
	//gamma = powertransform(gamma, 0.5);
	return gamma;

	
}

//void imgtest() {
//	//cv::Mat input1 = cv::imread("UnderwaterImg//最终结果.bmp", 0);
//	//medianBlur(input1, input1, 3);
//	//medianBlur(input1, input1, 3);
//	////medianBlur(input1, input1, 7);
//	//GaussianBlur(input1, input1, cv::Size(3, 3), 0, 0);
//	////GaussianBlur(input1, input1, cv::Size(5, 5), 0, 0);
//	//cv::imwrite("UnderwaterImg//最终结果滤波.bmp", input1);
//	cv::Mat input = cv::imread("UnderwaterImg//18.4.bmp", 0);
//	ImageProcess uwimg;
//	cv::Mat result1 = uwimg.DehazedImg(input);
//	cv::imwrite("UnderwaterImg//result1.bmp", result1);
//
//	cv::Mat result2 = uwimg.SPDF(result1);
//	cv::imwrite("UnderwaterImg//result2.bmp", result2);
//
//
//	char pathL[255], pathR[255];
//	sprintf(pathL, "UnderwaterImg//input//21.5//Camera00");
//	sprintf(pathR, "UnderwaterImg//input//21.5//Camera01");
//	string strL(pathL);
//	string strR(pathR);
//	vector<cv::String> pathLeft, pathRight;
//	cv::glob(strL, pathLeft, false);
//	cv::glob(strR, pathRight, false);
//	for (int i = 0; i < pathLeft.size(); i++) {
//		cv::Mat imgL = cv::imread(pathLeft[i], 0);
//		cv::Mat L1 = uwimg.DehazedImg(imgL);
//		cv::Mat resultL = uwimg.SPDF(L1);
//
//		string lpath = "UnderwaterImg//output//21.5//Camera00//" + to_string(i) + ".bmp";
//		cv::imwrite(lpath, resultL);
//
//		cv::Mat imgR = cv::imread(pathRight[i], 0);
//		cv::Mat R1 = uwimg.DehazedImg(imgR);
//		cv::Mat resultR = uwimg.SPDF(R1);
//
//		string rpath = "UnderwaterImg//output//21.5//Camera01//" + to_string(i) + ".bmp";
//		cv::imwrite(rpath, resultR);
//	}
//}

int main()
{
	ImageProcess uwimg;
	string strL = "C:\\Users\\11634\\Desktop\\水下结构光三维测量\\data\\input\\15.7\\Camera00\\";
	vector<cv::String> pathAll;
	cv::glob(strL, pathAll, false);
	for (int i = 0; i < pathAll.size(); i++) {
		cv::Mat imgL = cv::imread(pathAll[i], 0);
		
		clock_t start = clock();
		cv::Mat L1 = uwimg.DehazedImg(imgL);
		clock_t end = clock();
		cout << "花费了" << (double)(end - start) / CLOCKS_PER_SEC << "秒" << endl;

	}
	return 0;
}


int main0()
{
	ImageProcess uwimg;
	string strL = "C:\\Users\\11634\\Desktop\\水下结构光三维测量\\data\\input\\15.7\\Camera00\\";
	vector<cv::String> pathAll;
	cv::glob(strL, pathAll, false);
	for (int i = 0; i < pathAll.size(); i++) {
		cv::Mat imgL = cv::imread(pathAll[i], 0);
		cv::Mat L1 = uwimg.DehazedImg(imgL);
		cv::Mat resultL = uwimg.SPDF(L1);

		string dehaze_path = "C:\\Users\\11634\\Desktop\\水下结构光三维测量\\result\\dehazed\\" + to_string(i) + ".bmp";
		string final_path = "C:\\Users\\11634\\Desktop\\水下结构光三维测量\\result\\final\\" + to_string(i) + ".bmp";
		cv::imwrite(final_path, resultL);
		cv::imwrite(dehaze_path, L1);

		
	}
	return 0;
}