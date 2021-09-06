#!/usr/bin/env python

import os
# import sys
# sys.path.append('/home/bernard/.local/lib/python3.7/site-packages')
import cv2
import numpy as np
import scipy.ndimage
import math

global width, height
# diagonal length of img
global L
# spacing of a ray (used to calculate neighbor pixel)
phi = np.pi/12
# the threshold for L needed for N_r 
tau_e = 0.01
# threshold on the orientation similarity between p and l_vp
tau_p = 1 # np.pi # np.pi / 36
# angle of a ray
theta_r = 200*np.pi/180
# number of rays 
num_ray = 29
# angle to the next ray
ray_angle_diff = 5*np.pi/180


def image_grad(img):
    # res = cv2.GaussianBlur(img, (5, 5), 0)
    # grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    # grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

    # scaled_sobel_x = np.uint(255*abs(grad_x)) / np.max(abs(grad_x))
    # scaled_sobel_y = np.uint(255*abs(grad_y)) / np.max(abs(grad_y))
    # sx_binary = np.zeros_like(scaled_sobel_x)
    # sy_binary = np.zeros_like(scaled_sobel_y)

    # sx_binary = np.zeros_like(grad_x)
    # sy_binary = np.zeros_like(grad_y)

    # sx_binary[(scaled_sobel_x >= 30) & 
    #         (scaled_sobel_x <= 255)] = 1
    # sy_binary[(scaled_sobel_y >= 30) & 
    #         (scaled_sobel_y <= 255)] = 1 
    
    # # for debug
    # cv2.imshow('x', sx_binary)
    # cv2.imshow('y', sy_binary)
    
    # return sx_binary, sy_binary

    vertical_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    horizontal_filter = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    img = cv2.GaussianBlur(img, (5, 5), 0)
    x_grad = scipy.ndimage.correlate(img, horizontal_filter)
    y_grad = scipy.ndimage.correlate(img, vertical_filter)

    return x_grad, y_grad

def color_tensor_element(Blue_x, Blue_y, Green_x, Green_y, Red_x, Red_y, x_dir, y_dir):
    res = None
    if x_dir == True and y_dir == False:
        res = np.multiply(Blue_x, Blue_x) + np.multiply(Green_x, Green_x) + np.multiply(Red_x, Red_x) 
        # res = np.add(np.add(np.multiply(Blue_x, Blue_x), np.multiply(Green_x, Green_x)), np.multiply(Red_x, Red_x))

    elif x_dir == False and y_dir == True:
        res = np.multiply(Blue_y, Blue_y) + np.multiply(Green_y, Green_y) + np.multiply(Red_y, Red_y) 
        # res = np.add(np.add(np.multiply(Blue_y, Blue_y), np.multiply(Green_y, Green_y)), np.multiply(Red_y, Red_y))
        
    elif x_dir == True and y_dir == True:
        res = np.multiply(Blue_x, Blue_y) + np.multiply(Green_x, Green_y) + np.multiply(Red_x, Red_y)
        # res = np.add(np.add(np.multiply(Blue_x, Blue_y), np.multiply(Green_x, Green_y)), np.multiply(Red_x, Red_y))
 
    return res

def calc_vanishing_point(coordinates, local_orientation):
    """
    Calculation for the vanishing point
    """
    global width, height, L
    
    vp_score_max = 0
    from tqdm import tqdm
    for y_ind in tqdm(range(height)):
        for x_ind in range(width):
            vp_score_sum = 0
            for i, j in coordinates:
                l_vp = math.atan2(y_ind - j, x_ind - i) 
                # l_vp_list.append(l_vp)
                delta_vp = abs(local_orientation[y_ind, x_ind] -  l_vp)
                # delta_vp_list.append(delta_vp)
                mu_vp = math.sqrt((y_ind - j)**2 + (x_ind - i)**2) / L
                if delta_vp <= tau_p:
                    vp_score = math.exp(-delta_vp * mu_vp)
                else:
                    vp_score = 0
                vp_score_sum += vp_score
            # vp_score_max = max(vp_score_sum, vp_score_max)
            if vp_score_max < vp_score_sum:
                vp_score_max = vp_score_sum
                vanishing_point = (y_ind, x_ind)
            else:
                vp_score_max = vp_score_max
    return vanishing_point


def calc_d_o(img, local_orientation, vanishing_point, ray_list):
    """
    Calculation for d_o (orientation difference) 
    """
    global L
    # list of neighboring pixels of each ray 
    N_r_list = []
    R_list = []
    norm = np.linalg.norm
    vanishing_point = np.asarray(vanishing_point)
    ray_angle = theta_r
    for i in range(len(ray_list)):
        R_plus = []
        R_minus = []
        N_r = []
        for y in range(vanishing_point[1], height):
            for x in range(width):
                # p = np.asarray([x, y])
                # dist = abs(np.cross(ray_list[i] - vanishing_point, p - vanishing_point) / norm(ray_list[i] - vanishing_point))
                # if dist <= L*tau_e:
                #     N_r.append((x, y))
                vector_1 = (x, y) - vanishing_point
                vector_2 = ray_list[i] - vanishing_point
                # angle = np.arccos(np.dot(vector_1 / np.linalg.norm(vector_1), vector_2 / np.linalg.norm(vector_2)))
                angle = np.arctan2(vector_1[1], vector_1[0]) - np.arctan2(vector_2[1], vector_2[0])
                if -phi <= angle < 0:
                    R_minus.append([x, y])
                    N_r.append([x, y])
                elif 0 < angle <= ray_angle + phi:
                    R_plus.append([x, y])
                    N_r.append([x, y])
        R_list.append([R_plus, R_minus])
        N_r_list.append(N_r)

    ray_angle = theta_r
    d_o_list = []
    for i in range(len(N_r_list)):
        d_o = 0
        for j in range(len(N_r_list[i])):
            x = N_r_list[i][j][0]
            y = N_r_list[i][j][1]
            d_o += abs(ray_angle - local_orientation[y, x])
        ray_angle += ray_angle_diff
        d_o_list.append(d_o)

    return R_list, N_r_list, d_o_list

def main():
    global width, height, L
    # file_dir = "/home/bernard/Akrobotix/code/picture/"
    # data = [i for i in os.listdir(file_dir) if i.endswith('_pred.png')]
    # print(len(data))
    # test_pic = cv2.imread("/home/bernard/Akrobotix/code/picture/0.png")
    # img = cv2.imread("/home/bernard/Akrobotix/code/0.png")
    img = cv2.imread("0.png")
    scale_percent = 30  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    L = math.sqrt(height**2 + width**2)
    img = cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)

    Blue, Green, Red = cv2.split(img.astype("float"))

    Blue_x, Blue_y = image_grad(Blue)
    Green_x, Green_y = image_grad(Green)
    Red_x, Red_y = image_grad(Red)
    
    G_xx = color_tensor_element(Blue_x, Blue_y, Green_x, Green_y, Red_x, Red_y, True, False)
    G_yy = color_tensor_element(Blue_x, Blue_y, Green_x, Green_y, Red_x, Red_y, False, True)
    G_xy = color_tensor_element(Blue_x, Blue_y, Green_x, Green_y, Red_x, Red_y, True, True)
    local_orientation = 0.5*np.arctan2(2*G_xy, G_xx - G_yy) + (np.pi / 2)
    edge_strength = 0.5*(G_xx + G_yy + np.sqrt(np.square(G_xx - G_yy) + 4*np.square(G_xy)))

    edge_strength = np.uint(255*abs(edge_strength)) / np.max(abs(edge_strength))

    edge_strength = np.uint8(edge_strength)
    edges = cv2.Canny(edge_strength, 30, 200)
    # cv2.imshow('e_s', edge_strength)
    # cv2.imshow('edge', edges)
    # cv2.waitKey(1)
    # cv2.destroyAllWindows()
    indices = np.nonzero(edges)
    coordinates = list(zip(indices[1], indices[0]))
    
    # vanishing_point = calc_vanishing_point(coordinates, local_orientation, width, height)
    # print('---'*10)
    # print(vanishing_point)
    # print('---'*10)
    vanishing_point = (152, 40)
    image = cv2.circle(img, vanishing_point, radius=0, color=(0, 0, 255), thickness=10)
    # image = cv2.line(img, (0, vanishing_point[1]), (width-1, vanishing_point[1]), color=(255, 0, 0), thickness=3)
    
    j = 0
    ray_list = np.empty([num_ray, 2])
    for i in range(200, 341, 5):
        x = int(vanishing_point[0] + 200*math.cos(i*np.pi/180))
        y = int(vanishing_point[1] - 200*math.sin(i*np.pi/180))
        ray_list[j] = [x, y]
        j += 1
        image = cv2.line(img, (vanishing_point[0], vanishing_point[1]), (x, y), color=(0,255,0), thickness=1)
    
    R_list, N_r_list, d_o_list = calc_d_o(img, local_orientation, vanishing_point, ray_list)

    """
    Calculation for d_c (color difference)
    """
    color_seg_img = cv2.imread("0_pred.png")
    color_seg_img = cv2.resize(color_seg_img, (width, height), interpolation = cv2.INTER_AREA)
    # avg_color_per_row = np.average(color_seg_img, axis=0)
    # avg_color = np.average(avg_color_per_row, axis=0)
    d_c_list = []
    for i in range(len(R_list)):
        d_c = []
        c_plus = 0
        c_minus = 0
        for j in range(len(R_list[i])):
            plus_x = R_list[i][0][0]
            plus_y = R_list[i][0][1]
            minus_x = R_list[i][1][0]
            minus_y = R_list[i][1][1]
            c_plus += color_seg_img[plus_y, plus_x]
            c_minus += color_seg_img[minus_y, minus_x]
        c_plus /= len(R_list[i][0])    
        c_minus /= len(R_list[i][1])
        d_c.append([c_plus, c_minus])    
        d_c_list.append(d_c)
    print("done")
    # cv2.waitKey(0)
    cv2.imshow('img', image)
    # cv2.imwrite('vanishing_point.jpg', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("done")
    # cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()