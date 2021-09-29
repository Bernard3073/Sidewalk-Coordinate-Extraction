#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <string>
#include <math.h>

int width;
int height;
float L;
float tau_o = CV_PI/2;


void sanityCheck(cv::Mat checkMatrix);

void image_grad(cv::Mat img, cv::Mat &img_x, cv::Mat &img_y){

  // float sobelx_data[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
  // float sobely_data[9] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
  // cv::Mat sobelx = cv::Mat(3, 3, CV_32F, sobelx_data);
  // cv::Mat sobely = cv::Mat(3, 3, CV_32F, sobely_data);

  cv::GaussianBlur(img, img, cv::Size(5, 5), 0);
  // filter2D(img, img_x, -1, sobelx);
  // filter2D(img, img_y, -1, sobely);
  cv::Sobel(img, img_x, CV_64F, 1, 0, 5);
  cv::Sobel(img, img_y, CV_64F, 0, 1, 5);
  // cv::imshow("r", img_x);
  // cv::waitKey(0);
  // cv::destroyAllWindows();
}

void color_tensor(cv::Mat &res, cv::Mat Blue_x, cv::Mat Blue_y, cv::Mat Green_x, cv::Mat Green_y, cv::Mat Red_x, cv::Mat Red_y, bool x_dir, bool y_dir){
  if(x_dir && !y_dir){
    res = Blue_x.mul(Blue_x) + Green_x.mul(Green_x) + Red_x.mul(Red_x);
  }
  else if(!x_dir && y_dir){
    res = Blue_y.mul(Blue_y) + Green_y.mul(Green_y) + Red_y.mul(Red_y);
  }
  else if(x_dir && y_dir){
    res = Blue_x.mul(Blue_y) + Green_x.mul(Green_y) + Red_x.mul(Red_y);
  }
  // res = output;
}

// For Debug
// https://stackoverflow.com/questions/10167534/how-to-find-out-what-type-of-a-cv::mat-object-is-with-cv::mattype-in-opencv
std::string type2str(int type) {
  std::string r;
  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

void sanityCheck(cv::Mat checkMatrix)
{
  std::cout << std::endl << "Matrix holds:" <<std::endl;
  for (int r = 0; r < checkMatrix.rows; r++)
  {
    for (int c = 0; c < checkMatrix.cols; c++)
    {
      // unsigned char is an unsigned byte value(0-255)
      std::cout << checkMatrix.at<double>(r, c) << " | ";
    }
    std::cout << std::endl;
  }
}


void calc_vanishing_point(cv::Point &vanishing_point, std::vector<cv::Point> edge_coord, cv::Mat local_orientation){
  float vp_score_max = 0.0, vp_score_sum = 0.0;
  float l_vp, delta_vp, mu_vp, vp_score;
  for(int y_ind = 0; y_ind < height; y_ind++){
    for(int x_ind = 0; x_ind < width; x_ind++){
      vp_score_sum = 0.0;
      for(int edge = 0; edge < edge_coord.size(); edge++){
        int i = edge_coord[edge].x; 
        int j = edge_coord[edge].y; 
        if(y_ind <= j) continue;
        // l_vp = atan2(y_ind - j, x_ind - i);
        float dot = i*x_ind + j*y_ind;
        float det = i*y_ind - j*x_ind;
        l_vp = atan2(det, dot);
        delta_vp = abs(local_orientation.at<double>(y_ind, x_ind) - l_vp);
        if(delta_vp <= tau_o){
          mu_vp = sqrt(pow(y_ind - j, 2) + pow(x_ind - i, 2)) / L;
          vp_score = exp(-delta_vp * mu_vp);
        }
        else vp_score = 0.0;
        vp_score_sum += vp_score;
      }
      // if(vp_score_sum != 0) std::cout << vp_score_sum << '\n';
      if(vp_score_max < vp_score_sum){
        vp_score_max = vp_score_sum;
        vanishing_point.x = x_ind;
        vanishing_point.y = y_ind;
        std::cout << vanishing_point.x << ", " << vanishing_point.y << '\n';
      }
      else vp_score_max = vp_score_max;
    }
    // if(vp_score_max != 0) std::cout << vp_score_max << '\n';
  }
}

int main( int argc, char** argv ) {
  
  cv::Mat image;
  image = cv::imread("frame002730.jpg.png");
  int scale_percent = 30;
  width = image.cols * scale_percent / 100;
  height = image.rows * scale_percent / 100;
  cv::resize(image, image, cv::Size(width, height), cv::INTER_AREA);
  L = sqrt(width^2 + height^2);

  if(! image.data ) {
      std::cout <<  "Image not found or unable to open" << std::endl ;
      return -1;
  }
  // cv::Mat three_channels[3];
  // cv::split(image, three_channels);
  cv::Mat three_channels[3]; 
  cv::split(image, three_channels);
  cv::Mat Blue = cv::Mat::zeros(cv::Size(image.cols+1, image.rows+1), CV_32F);
  cv::Mat Green = cv::Mat::zeros(cv::Size(image.cols+1, image.rows+1), CV_32F);
  cv::Mat Red = cv::Mat::zeros(cv::Size(image.cols+1, image.rows+1), CV_32F);
  for (int r = 0; r < image.rows; r++)
  {
    for (int c = 0; c < image.cols; c++)
    {
      // Blue.at<uchar>(r, c) = image.at<cv::Vec3b>(r, c)[0];
      // Green.at<uchar>(r, c) = image.at<cv::Vec3b>(r, c)[1];
      // Red.at<uchar>(r, c) = image.at<cv::Vec3b>(r, c)[2];
      Blue.at<float>(r, c) = three_channels[0].at<uchar>(r, c);
      Green.at<float>(r, c) = three_channels[1].at<uchar>(r, c);
      Red.at<float>(r, c) = three_channels[2].at<uchar>(r, c);
    }
    // std::cout << std::endl;
  }

  cv::Mat Blue_x, Blue_y, Green_x, Green_y, Red_x, Red_y;
  image_grad(Blue, Blue_x, Blue_y);
  image_grad(Green, Green_x, Green_y);
  image_grad(Red, Red_x, Red_y);

  // cv::Mat G_xx = cv::Mat::zeros(cv::Size(Blue_x.rows, Blue_x.cols), CV_32F);
  // cv::Mat G_yy = cv::Mat::zeros(cv::Size(Blue_x.rows, Blue_x.cols), CV_32F);
  // cv::Mat G_xy = cv::Mat::zeros(cv::Size(Blue_x.rows, Blue_x.cols), CV_32F);
  cv::Mat G_xx, G_yy, G_xy;
  color_tensor(G_xx, Blue_x, Blue_y, Green_x, Green_y, Red_x, Red_y, true, false);
  color_tensor(G_yy, Blue_x, Blue_y, Green_x, Green_y, Red_x, Red_y, false, true);
  color_tensor(G_xy, Blue_x, Blue_y, Green_x, Green_y, Red_x, Red_y, true, true);

  // For Debug
  // std::string ty =  type2str( G_xy.type() );
  // std::string ty2 =  type2str( xx_sub_yy.type() );
  
  cv::Mat tensor_angle; 
  cv::Mat xx_sub_yy;
  xx_sub_yy = G_xx - G_yy;
  // phase(xx_sub_yy, 2*G_xy, tensor_angle);
  // add(0.5*tensor_angle, CV_PI / 2, local_orientation);
  // phase(xx_sub_yy, 2*G_xy, local_orientation);
  // sanityCheck(G_xx);
  // sanityCheck(G_yy);
  // sanityCheck(G_xy);

  G_xx.convertTo(G_xx, CV_32F);
  G_yy.convertTo(G_yy, CV_32F);
  G_xy.convertTo(G_xy, CV_32F);

  cv::Mat local_orientation;
  phase(G_xx - G_yy, 2*G_xy, local_orientation);
  
  // cv::Mat local_orientation = cv::Mat::zeros(cv::Size(G_xy.rows, G_xy.cols), CV_32F);
  // for (int r = 0; r < G_xy.rows; r++)
  // {
  //   for (int c = 0; c < G_xy.cols; c++)
  //   {
  //     local_orientation.at<float>(r, c) =  atan2(2*G_xy.at<float>(r, c), xx_sub_yy.at<float>(r, c)) + (CV_PI/2);
  //     // std::cout << local_orientation.at<double>(r, c) << " | ";
  //   }
  //   // std::cout << std::endl;
  // }

  // local_orientation = 0.5*local_orientation + (CV_PI / 2);
  cv::Mat xx_yy_square, xy_square;
  pow(G_xx - G_yy, 2.0, xx_yy_square);
  pow(G_xy, 2.0, xy_square);
  cv::Mat sqrt_x_y, add_x_y, add_xx_yy;
  add(xx_yy_square, 4*xy_square, add_x_y);
  sqrt(add_x_y, sqrt_x_y);
  // cv::Mat edge_strength = cv::Mat::zeros(cv::Size(G_xy.rows, G_xy.cols), CV_32F);
  cv::Mat edge_strength;
  // for (int r = 0; r < G_xy.rows; r++)
  // {
  //   for (int c = 0; c < G_xy.cols; c++)
  //   {
  //     edge_strength.at<double>(r, c) = 0.5*(G_xx.at<double>(r, c) + G_yy.at<double>(r, c) + sqrt(pow(xx_sub_yy.at<double>(r, c), 2.0) + 4*pow(G_xy.at<double>(r, c), 2.0)));
  //     // std::cout << edge_strength.at<double>(r, c) << " | ";
  //   }
  //   // std::cout << std::endl;
  // }
  edge_strength = 0.5*(G_xx + G_yy + sqrt_x_y);
  double minVal, maxVal;
  minMaxLoc(edge_strength, &minVal, &maxVal);
  cv::Mat res;
  edge_strength = 255*edge_strength / maxVal;
  edge_strength.convertTo(edge_strength, CV_8UC1);
  cv::Mat canny;
  cv::Canny(edge_strength, canny, 0, 200);
  // cv::imshow("e", edge_strength);
  // imshow("c", canny);
  // cv::waitKey(0);
  // cv::destroyAllWindows();
  std::vector<std::vector<cv::Point> > contours;
  std::vector<cv::Vec4i> hierarchy;

  findContours(canny, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
  cv::Point vanishing_point(0, 0);
  std::vector<cv::Point> edge_coord;
  for(int i=0; i<contours.size(); i++){
    for(int j=0; j<contours[i].size(); j++){
      edge_coord.push_back(contours[i][j]);
    }
  }
  calc_vanishing_point(vanishing_point, edge_coord, local_orientation);
  cv::circle(image, vanishing_point, 0, cv::Scalar(0,0,255), 10);
  cv::imshow("img", image);
  cv::waitKey(0);
  cv::destroyAllWindows();
  return 0;
}