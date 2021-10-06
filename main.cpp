#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <string>
#include <math.h>
#include <algorithm>

unsigned int width;
unsigned int height;
float L;
// threshold on the orientation similarity between p and l_vp
float tau_o = CV_PI/2;
// spacing of a ray (used to calculate neighbor pixel)
float phi = CV_PI/12;
// angle of a ray
float theta_r = 200*CV_PI/180;
// number of rays
unsigned int num_ray = 29;
// angle to the next ray
double ray_angle_diff = 5*CV_PI/180;
// number of bins in the color histogram
unsigned int bins = 16;

float a = 0.9;
float b = 2.3;
float alpha = 0.9;
float beta = 2.3;
float sigma = 0.3437;
float phi_avg = 1.597;

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
      std::cout << checkMatrix.at<float>(r, c) << " | ";
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
        // std::cout << vanishing_point.x << ", " << vanishing_point.y << '\n';
      }
      else vp_score_max = vp_score_max;
    }
    // if(vp_score_max != 0) std::cout << vp_score_max << '\n';
  }
}

float calc_u_ij(cv::Mat image, cv::Mat color_seg_img, cv::Point &vanishing_point, cv::Point &ray_i, cv::Point &ray_j){
  // Calculate color uniformity between ray "i" and ray "j"
  std::vector<cv::Point> left_pts, right_pts;
  // // cv::Mat left_mask(color_seg_img.size(), uchar(0));
  // cv::Mat right_mask(color_seg_img.size(), uchar(0));
  // cv::clipLine(color_seg_img.size(), vanishing_point, ray_i);
  // // cv::clipLine(color_seg_img.size(), vanishing_point, ray_j);
  // // cv::line(left_mask, vanishing_point, ray_i,  cv::Scalar(255));
  // cv::line(right_mask, vanishing_point, ray_j, cv::Scalar(255));
  // // cv::findNonZero(left_mask, left_pts);
  // cv::findNonZero(right_mask, right_pts);

  // cv::LineIterator it_left(image, vanishing_point, ray_i, 8);
  // cv::LineIterator it_right(image, vanishing_point, ray_j, 8);
  // for(int i=0; i<it_left.count; i++, ++it_left){
  //   cv::Point pt = it_left.pos();
  //   left_pts.push_back(pt);
  // }
  // for(int i=0; i<it_right.count; i++, ++it_right){
  //   cv::Point pt = it_right.pos();
  //   right_pts.push_back(pt);
  // }

  std::vector<cv::Point> ray_area_pts;
  // cv::Rect left_rect = cv::boundingRect(left_pts);
  // cv::Rect right_rect = cv::boundingRect(right_pts);
  // cv::Point ray_area_x_min = left_rect.tl();
  // cv::Point ray_area_x_max = left_rect.br() - cv::Point(1, 1);
  // cv::line(image, left_rect.tl(), left_rect.br() - cv::Point(1, 1), cv::Scalar(0,255,0), 1);
  // cv::imshow("img", image);
  // cv::waitKey(0);
  // cv::destroyAllWindows();

  double m1 = double(ray_i.y - vanishing_point.y) / double( ray_i.x - vanishing_point.x);
  double m2 = double(ray_j.y - vanishing_point.y) / double(ray_j.x - vanishing_point.x);
  double m_min = std::min(m1, m2);
  double m_max = std::max(m1, m2);
  for(int y=vanishing_point.y; y<height; y++){
    for(int x=0; x<width; x++){
      cv::Point cur_pt(x, y);
      double m;
      if(cur_pt.x != vanishing_point.x) m = (cur_pt.y - vanishing_point.y) / ( cur_pt.x - vanishing_point.x);
      else m = 0;
      if(m >= m_min && m <= m_max) ray_area_pts.push_back(cur_pt);
    }
  }
  // std::vector<cv::Point> ray_area_pts;
  // for(int i=0; i<left_pts.size(); i++){
  //   ray_area_pts.push_back(left_pts[i]);
  //   ray_area_pts.push_back(right_pts[i]);
  // }
  cv::Mat ray_area_mask = cv::Mat::zeros(color_seg_img.size(), CV_8UC1);
  cv::fillPoly(ray_area_mask, ray_area_pts, cv::Scalar(255));
  cv::Mat color_seg_img_copy = cv::Mat::zeros(image.size(), image.type());
  // std::string ty =  type2str(test.type() );
  // std::string ty1 =  type2str( image.type() );
  // std::string ty2 =  type2str( ray_area_mask.type() );
  
  cv::bitwise_and(color_seg_img, color_seg_img, color_seg_img_copy, ray_area_mask);
  // image.copyTo(test, ray_area_mask);
  // cv::imshow("img", color_seg_img_copy);
  // cv::imshow("img1", ray_area_mask);
  // cv::waitKey(0);
  // cv::destroyAllWindows();
  
  cv::Mat three_channels[3]; 
  cv::split(color_seg_img, three_channels);
  // cv::Mat Blue = cv::Mat::zeros(cv::Size(color_seg_img.cols+1, color_seg_img.rows+1), CV_32F);
  // cv::Mat Green = cv::Mat::zeros(cv::Size(color_seg_img.cols+1, color_seg_img.rows+1), CV_32F);
  // cv::Mat Red = cv::Mat::zeros(cv::Size(color_seg_img.cols+1, color_seg_img.rows+1), CV_32F);
  // for (int r = 0; r < color_seg_img.rows; r++)
  // {
  //   for (int c = 0; c < color_seg_img.cols; c++)
  //   {
  //     Blue.at<float>(r, c) = three_channels[0].at<uchar>(r, c);
  //     Green.at<float>(r, c) = three_channels[1].at<uchar>(r, c);
  //     Red.at<float>(r, c) = three_channels[2].at<uchar>(r, c);
  //   }
  // }
  // three_channels[0].convertTo(three_channels[0], CV_8UC3);
  // sanityCheck(three_channels[0]);
  int histSize = 16;

  float range[] = { 0, 256 } ;
  const float* histRange = { range };

  bool uniform = true; 
  bool accumulate = false;

  cv::Mat b_hist, g_hist, r_hist;
  
  cv::calcHist( &three_channels[0], 1, 0, cv::Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
  cv::calcHist( &three_channels[1], 1, 0, cv::Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
  cv::calcHist( &three_channels[2], 1, 0, cv::Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );
  cv::normalize(b_hist, b_hist, 0, 255, cv::NORM_MINMAX);
  cv::normalize(g_hist, g_hist, 0, 255, cv::NORM_MINMAX);
  cv::normalize(r_hist, r_hist, 0, 255, cv::NORM_MINMAX);
  // sanityCheck(b_hist); // float
  cv::Scalar b_hist_sum = cv::sum(b_hist);
  cv::Scalar g_hist_sum = cv::sum(g_hist);
  cv::Scalar r_hist_sum = cv::sum(r_hist);
  // cv::divide(b_hist, cv::sum());
  cv::Scalar u;
  cv::Mat b_hist_square, g_hist_square, r_hist_square;
  cv::pow(b_hist/b_hist_sum[0], 2, b_hist_square);
  cv::pow(g_hist/g_hist_sum[0], 2, g_hist_square);
  cv::pow(r_hist/r_hist_sum[0], 2, r_hist_square);
  u = cv::sum(b_hist_square) + cv::sum(g_hist_square) + cv::sum(r_hist_square);
  std::cout << u << '\n';

  return u[0];
}

// float lane_score(std::vector<double> d_o_list, std::vector<float> d_c_list, float uniformity, std::vector<int> phi_list){

// }


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
  cv::Mat three_channels[3]; 
  cv::split(image, three_channels);
  cv::Mat Blue = cv::Mat::zeros(cv::Size(image.cols+1, image.rows+1), CV_32F);
  cv::Mat Green = cv::Mat::zeros(cv::Size(image.cols+1, image.rows+1), CV_32F);
  cv::Mat Red = cv::Mat::zeros(cv::Size(image.cols+1, image.rows+1), CV_32F);
  for (int r = 0; r < image.rows; r++)
  {
    for (int c = 0; c < image.cols; c++)
    {
      Blue.at<float>(r, c) = three_channels[0].at<uchar>(r, c);
      Green.at<float>(r, c) = three_channels[1].at<uchar>(r, c);
      Red.at<float>(r, c) = three_channels[2].at<uchar>(r, c);
    }
  }

  cv::Mat Blue_x, Blue_y, Green_x, Green_y, Red_x, Red_y;
  image_grad(Blue, Blue_x, Blue_y);
  image_grad(Green, Green_x, Green_y);
  image_grad(Red, Red_x, Red_y);

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

  G_xx.convertTo(G_xx, CV_32F);
  G_yy.convertTo(G_yy, CV_32F);
  G_xy.convertTo(G_xy, CV_32F);

  cv::Mat local_orientation;
  phase(G_xx - G_yy, 2*G_xy, local_orientation);
  cv::Mat xx_yy_square, xy_square;
  pow(G_xx - G_yy, 2.0, xx_yy_square);
  pow(G_xy, 2.0, xy_square);
  cv::Mat sqrt_x_y, add_x_y, add_xx_yy;
  add(xx_yy_square, 4*xy_square, add_x_y);
  sqrt(add_x_y, sqrt_x_y);
  cv::Mat edge_strength;
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
  // cv::imshow("img", image);
  // cv::waitKey(0);
  // cv::destroyAllWindows();


  // list of ray angle
  std::vector<int> phi_list;
  // list of the end point of each ray
  std::vector<cv::Point> ray_coord_list;
  // store the rays' end points in a dynamic array
  for(int i=200; i<341; i+=5){
    phi_list.push_back(i*CV_PI/180);
    cv::Point ray_end_point;
    ray_end_point.x = int(vanishing_point.x + 200*cos(i*CV_PI/180));
    ray_end_point.y = int(vanishing_point.y - 200*sin(i*CV_PI/180));
    if(ray_end_point.x < 0) ray_end_point.x = 0;
    else if(ray_end_point.x > width) ray_end_point.x = width;
    else if(ray_end_point.y > height) ray_end_point.y = height;
    ray_coord_list.push_back(ray_end_point);
    // cv::line(image, vanishing_point, ray_end_point, cv::Scalar(0,255,0), 1);
  }
  // cv::imshow("img", image);
  // cv::waitKey(0);
  // cv::destroyAllWindows();
  
  std::vector<std::pair<std::vector<cv::Point>, std::vector<cv::Point>>>R_list;
  // list of neighbor pixels of each ray
  std::vector<std::vector<cv::Point>> N_R_list;
  double ray_angle = theta_r;
  cv::Point vector_1, vector_2;
  double angle;
  // calculate orientation difference
  for(int i=0; i<ray_coord_list.size(); i++){
    std::vector<cv::Point> R_plus;
    std::vector<cv::Point> R_minus;
    std::vector<cv::Point> N_r;
    for(int y=vanishing_point.y; y<height; y++){
      for(int x=0; x<width; x++){
        cv::Point cur_pt;
        vector_1.x = x - vanishing_point.x;
        vector_1.y = y - vanishing_point.y;
        vector_2.x = ray_coord_list[i].x - vanishing_point.x;
        vector_2.y = ray_coord_list[i].y - vanishing_point.y;
        angle = double(atan2(vector_1.y, vector_1.x)) - double(atan2(vector_2.y, vector_2.x));
        cur_pt.x = x;
        cur_pt.y = y;
        if(angle >= -phi && angle < 0){
          R_minus.push_back(cur_pt);
          N_r.push_back(cur_pt);
        }
        else if(angle > 0 && angle <= ray_angle + phi){
          R_plus.push_back(cur_pt);
          N_r.push_back(cur_pt);
        }
      }
    }
    R_list.push_back(std::make_pair(R_plus, R_minus));
    N_R_list.push_back(N_r);
  }
  double d_o;
  std::vector<double> d_o_list;
  cv::Point cur;
  for(int i=0; i<N_R_list.size(); i++){
    d_o = 0.0;
    for(int j=0; j<N_R_list[i].size(); j++){
      cur.x = N_R_list[i][j].x;
      cur.y = N_R_list[i][j].y;
      d_o += abs(ray_angle - local_orientation.at<double>(cur.y, cur.x));
    }
    ray_angle += ray_angle_diff;
    d_o_list.push_back(d_o / N_R_list[i].size());
  }

  // Calculate color difference
  cv::Mat color_seg_img = cv::imread("frame002730.jpg_pred.png");
  cv::resize(color_seg_img, color_seg_img, cv::Size(width, height), cv::INTER_AREA);
  // store the color difference of each ray in a vector
  std::vector<float> d_c_list;
  for(int i=0; i<R_list.size(); i++){
    
    std::vector<int> c_plus{0, 0, 0};
    for(int j=0; j<R_list[i].first.size(); j++){
      cv::Vec3b bgrPixel = color_seg_img.at<cv::Vec3f>(R_list[i].first[j].x, R_list[i].first[j].y);
      uchar blue = bgrPixel[0];
      uchar green = bgrPixel[1];
      uchar red = bgrPixel[2]; 
      c_plus[0] += blue;
      c_plus[1] += green;
      c_plus[2] += red;
    }
    for(int k=0; k<2; k++) c_plus[k] /= R_list[i].first.size();

    std::vector<int> c_minus{0, 0, 0};
    for(int j=0; j<R_list[i].second.size(); j++){
      cv::Vec3b bgrPixel = color_seg_img.at<cv::Vec3f>(R_list[i].second[j].x, R_list[i].second[j].y);
      uchar blue = bgrPixel[0];
      uchar green = bgrPixel[1];
      uchar red = bgrPixel[2]; 
      c_minus[0] += blue;
      c_minus[1] += green;
      c_minus[2] += red;
    }
    for(int k=0; k<2; k++) c_minus[k] /= R_list[i].second.size();
    float d_c = sqrt(pow(c_plus[0] - c_minus[0], 2) + pow(c_plus[1] - c_minus[1], 2) + pow(c_plus[2] - c_minus[2], 2));
    //  / max(sqrt(sum(pow(c_plus, 2))), sqrt(sum(pow(c_minus, 2))))
    d_c = d_c / std::max(pow(c_plus[0], 2) + pow(c_plus[1], 2) + pow(c_plus[2], 2), pow(c_minus[0], 2) + pow(c_minus[1], 2) + pow(c_minus[2], 2));
    d_c_list.push_back(d_c);
  }
  for(int i=0; i<ray_coord_list.size()-1; i++){
    for(int j = i+1; j<ray_coord_list.size(); j++){
        float uniformity = calc_u_ij(image, color_seg_img, vanishing_point, ray_coord_list[i], ray_coord_list[j]);

    }
  }
  std::cout << "Done" << '\n';
  return 0;
}