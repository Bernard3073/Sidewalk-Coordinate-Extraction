#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

using namespace cv;

void image_grad(Mat img, Mat &img_x, Mat &img_y){

  float sobelx_data[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
  float sobely_data[9] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
  Mat sobelx = Mat(3, 3, CV_32F, sobelx_data);
  Mat sobely = Mat(3, 3, CV_32F, sobely_data);
  // Mat img_x, img_y;
  filter2D(img, img_x, -1, sobelx);
  filter2D(img, img_y, -1, sobely);

  // imshow("r", img_x);
  // return img_x, img_y;
}

void color_tensor(Mat &res, Mat Blue_x, Mat Blue_y, Mat Green_x, Mat Green_y, Mat Red_x, Mat Red_y, bool x_dir, bool y_dir){
  Mat output;
  if(x_dir && !y_dir){
    // add(Blue_x.mul(Blue_x), Green_x.mul(Green_x), output);
    // std::cout << "output row: 0~2 = "<< std::endl << " "  << output.rowRange(0, 2) << std::endl << std::endl;
    // add(output, Red_x.mul(Red_x), res);
    // std::cout << "output row: 0~2 = "<< std::endl << " "  << output.rowRange(0, 2) << std::endl << std::endl;
    res = Blue_x.mul(Blue_x) + Green_x.mul(Green_x) + Red_x.mul(Red_x);
  }
  else if(!x_dir && y_dir){
    // add(Blue_y.mul(Blue_y), Green_y.mul(Green_y), output);
    // add(output, Red_y.mul(Red_y), res);
    res = Blue_y.mul(Blue_y) + Green_y.mul(Green_y) + Red_y.mul(Red_y);
  }
  else if(x_dir && y_dir){
    // add(Blue_x.mul(Blue_y), Green_x.mul(Green_y), output);
    // add(output, Red_x.mul(Red_y), res);
    res = Blue_x.mul(Blue_y) + Green_x.mul(Green_y) + Red_x.mul(Red_y);
  }
  // res = output;
}

// For Debug
// https://stackoverflow.com/questions/10167534/how-to-find-out-what-type-of-a-mat-object-is-with-mattype-in-opencv
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

void sanityCheck(Mat checkMatrix)
{
  std::cout << std::endl << "Matrix holds:" <<std::endl;
  for (int r = 0; r < 5; r++)
  {
    for (int c = 0; c < 5; c++)
    {
      std::cout << checkMatrix.at<double>(r, c) << " | ";
    }
    std::cout << std::endl;
  }
}

int main( int argc, char** argv ) {
  
  Mat image;
  image = imread("0.png" ,IMREAD_COLOR);
  
  if(! image.data ) {
      std::cout <<  "Image not found or unable to open" << std::endl ;
      return -1;
  }
  
  Mat rgbchannel[3];
  split(image, rgbchannel);
  Mat Red = rgbchannel[0];
  Mat Green = rgbchannel[1];
  Mat Blue = rgbchannel[2];

  // cv::imshow("red", Red);
  // cv::imshow("green", Green);
  // cv::imshow("blue", Blue);
  // waitKey(0);
  // destroyAllWindows();
  Mat Blue_x, Blue_y, Green_x, Green_y, Red_x, Red_y;
  Blue_x.convertTo(Blue_x, CV_8U);
  sanityCheck(image);
  image_grad(Blue, Blue_x, Blue_y);
  image_grad(Green, Green_x, Green_y);
  image_grad(Red, Red_x, Red_y);

  // imshow("B", Blue_x);
  Mat G_xx, G_yy, G_xy;
  color_tensor(G_xx, Blue_x, Blue_y, Green_x, Green_y, Red_x, Red_y, true, false);
  color_tensor(G_yy, Blue_x, Blue_y, Green_x, Green_y, Red_x, Red_y, false, true);
  color_tensor(G_xy, Blue_x, Blue_y, Green_x, Green_y, Red_x, Red_y, true, true);

  // std::cout << "G_xx row: 0~2 = "<< std::endl << " "  << G_xx.rowRange(0, 2) << std::endl << std::endl;
  // std::cout << "G_yy row: 0~2 = "<< std::endl << " "  << G_yy.rowRange(0, 2) << std::endl << std::endl;
  G_xx.convertTo(G_xx, CV_32F);
  G_yy.convertTo(G_yy, CV_32F);
  G_xy.convertTo(G_xy, CV_32F);

  // For Debug
  // std::string ty =  type2str( G_xy.type() );
  // std::string ty2 =  type2str( xx_sub_yy.type() );
  
  // local_orientation = 0.5*np.arctan2(2*G_xy, G_xx - G_yy) + (np.pi / 2)
  // edge_strength = 0.5*(G_xx + G_yy + np.sqrt(np.square(G_xx - G_yy) + 4*np.square(G_xy)))
  Mat tensor_angle, local_orientation, edge_strength; 
  Mat xx_sub_yy;
  // subtract(G_xx, G_yy, xx_sub_yy);
  // phase(xx_sub_yy, 2*G_xy, tensor_angle);
  // add(0.5*tensor_angle, CV_PI / 2, local_orientation);
  // phase(xx_sub_yy, 2*G_xy, local_orientation);
  sanityCheck(G_xx);
  phase(G_xx - G_yy, 2*G_xy, local_orientation);
  local_orientation = 0.5*local_orientation + (CV_PI / 2);
  Mat xx_yy_square, xy_square;
  pow(G_xx - G_yy, 2.0, xx_yy_square);
  pow(G_xy, 2.0, xy_square);
  Mat sqrt_x_y, add_x_y, add_xx_yy;
  add(xx_yy_square, 4*xy_square, add_x_y);
  sqrt(add_x_y, sqrt_x_y);
  // add(G_xx, G_yy, add_xx_yy);
  // add(add_xx_yy, sqrt_x_y, edge_strength);
  edge_strength = 0.5*(G_xx + G_yy + sqrt_x_y);

  // edge_strength = np.uint(255*abs(edge_strength)) / np.max(abs(edge_strength))
  // img_binary = np.zeros_like(edge_strength)
  // # setting threshold
  // img_binary[(edge_strength >= 30) & 
  //         (edge_strength <= 255)] = 1
  double minVal, maxVal;
  minMaxLoc(edge_strength, &minVal, &maxVal);
  Mat res;
  // normalize(edge_strength, res, minVal, maxVal, NORM_MINMAX);
  edge_strength = 255*edge_strength / maxVal;
  imshow("e", edge_strength);

  waitKey(0);
  destroyAllWindows();
  return 0;
}