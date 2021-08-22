#!/usr/bin/env python

import os
# import sys
# sys.path.append('/home/bernard/.local/lib/python3.7/site-packages')
import cv2
import numpy as np
import scipy.ndimage

# def convolution(img, kernel):
#     img_row, img_col = img.shape


def image_grad(img):
    # res = cv2.Gaussianres(img, (5, 5), 0)
    # grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    # grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

    # scaled_sobel_x = np.uint(255*abs(grad_x)) / np.max(abs(grad_x))
    # scaled_sobel_y = np.uint(255*abs(grad_y)) / np.max(abs(grad_y))
    # sx_binary = np.zeros_like(scaled_sobel_x)
    # sy_binary = np.zeros_like(scaled_sobel_y)

    # sx_binary = np.zeros_like(grad_x)
    # sy_binary = np.zeros_like(grad_y)

    # sx_binary[(scaled_sobel_x >= 100) & 
    #         (scaled_sobel_x <= 255)] = 1
    # sy_binary[(scaled_sobel_y >= 100) & 
    #         (scaled_sobel_y <= 255)] = 1 
    
    # # for debug
    # cv2.imshow('x', sx_binary)
    # cv2.imshow('y', sy_binary)
    
    # return sx_binary, sy_binary

    vertical_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    horizontal_filter = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, 1]])

    x_grad = scipy.ndimage.correlate(img, horizontal_filter)
    y_grad = scipy.ndimage.correlate(img, vertical_filter)

    return x_grad, y_grad

def color_tensor_element(B_x, B_y, G_x, G_y, R_x, R_y, x_dir, y_dir):
    res = None
    if x_dir == True and y_dir == False:
        res = np.multiply(B_x, B_x) + np.multiply(G_x, G_x) + np.multiply(R_x, R_x) 
        

    elif x_dir == False and y_dir == True:
        res = np.multiply(B_y, B_y) + np.multiply(G_y, G_y) + np.multiply(R_y, R_y) 
        
    elif x_dir == True and y_dir == True:
        res = np.multiply(B_x, B_y) + np.multiply(G_x, G_y) + np.multiply(R_x, R_y)

    res = cv2.GaussianBlur(res, (5, 5), 0)  
    return res

def main():
    # file_dir = "/home/bernard/Akrobotix/code/picture/"
    # data = [i for i in os.listdir(file_dir) if i.endswith('_pred.png')]
    # print(len(data))
    test_pic = cv2.imread("/home/bernard/Akrobotix/code/picture/0.png")
    
    # gray = cv2.cvtColor(test_pic, cv2.COLOR_BGR2GRAY)
    # res = cv2.Gaussianres(gray, (5, 5), 0)
    # grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    # grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    
    # B, G, R = cv2.split(test_pic.astype("float"))
    # B_x, B_y = image_grad(B)
    # canny = cv2.Canny(test_pic, 10, 100)
    # cv2.imshow('test', canny)

    B, G, R = cv2.split(test_pic.astype("float"))
    B_x, B_y = image_grad(B)
    G_x, G_y = image_grad(G)
    R_x, R_y = image_grad(R)
    tensor_xx = color_tensor_element(B_x, B_y, G_x, G_y, R_y, R_y, True, False)
    tensor_yy = color_tensor_element(B_x, B_y, G_x, G_y, R_y, R_y, False, False)
    tensor_xy = color_tensor_element(B_x, B_y, G_x, G_y, R_y, R_y, False, True)

    edge_strength = 0.5*(tensor_xx + tensor_yy + np.sqrt(np.square(tensor_xx - tensor_yy) + 4*np.square(tensor_xy)))
    cv2.imshow('e', edge_strength)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()