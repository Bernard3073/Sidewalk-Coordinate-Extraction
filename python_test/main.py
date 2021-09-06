#!/usr/bin/env python

import os
# import sys
# sys.path.append('/home/bernard/.local/lib/python3.7/site-packages')
import cv2
import numpy as np
import scipy.ndimage
import math
# def convolution(img, kernel):
#     img_row, img_col = img.shape


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

def cal_vanishing_point(coordinates, local_orientation, width, height):
    # diagonal length of img
    L = math.sqrt(height**2 + width**2)
    # threshold on the orientation similarity between p and l_vp
    tau_p = 1 # np.pi # np.pi / 36
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


def main():
    # file_dir = "/home/bernard/Akrobotix/code/picture/"
    # data = [i for i in os.listdir(file_dir) if i.endswith('_pred.png')]
    # print(len(data))
    # test_pic = cv2.imread("/home/bernard/Akrobotix/code/picture/0.png")
    # frame = cv2.imread("/home/bernard/Akrobotix/code/0.png")
    frame = cv2.imread("0.png")
    scale_percent = 30  # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    frame = cv2.resize(frame, (width, height), interpolation = cv2.INTER_AREA)

    # B, G, R = cv2.split(test_pic.astype("float"))
    # Blue_x, Blue_y = image_grad(B)
    # canny = cv2.Canny(test_pic, 50, 100)
    # cv2.imshow('test', canny)

    Blue, Green, Red = cv2.split(frame.astype("float"))

    Blue_x, Blue_y = image_grad(Blue)
    Green_x, Green_y = image_grad(Green)
    Red_x, Red_y = image_grad(Red)
    
    G_xx = color_tensor_element(Blue_x, Blue_y, Green_x, Green_y, Red_x, Red_y, True, False)
    G_yy = color_tensor_element(Blue_x, Blue_y, Green_x, Green_y, Red_x, Red_y, False, True)
    G_xy = color_tensor_element(Blue_x, Blue_y, Green_x, Green_y, Red_x, Red_y, True, True)
    local_orientation = 0.5*np.arctan2(2*G_xy, G_xx - G_yy) + (np.pi / 2)
    edge_strength = 0.5*(G_xx + G_yy + np.sqrt(np.square(G_xx - G_yy) + 4*np.square(G_xy)))

    # edge_strength = ((edge_strength - edge_strength.min()) / (edge_strength.max() - edge_strength.min()))*255
    # edge_strength = np.uint(edge_strength)

    edge_strength = np.uint(255*abs(edge_strength)) / np.max(abs(edge_strength))

    # img_binary = np.zeros_like(edge_strength)
    # setting threshold
    # img_binary[(edge_strength >= 30) & 
    #         (edge_strength <= 255)] = 1
    # cv2.imshow('t', img_binary)
    # Blue = np.uint8(Blue)
    # Green = np.uint8(Green)
    # Red = np.uint8(Red)
    # Blue_cny = cv2.Canny(Blue, 30, 200)
    # Green_cny = cv2.Canny(Green, 30, 200)
    # Red_cny = cv2.Canny(Red, 30, 200)
    # img_cny = cv2.merge([Blue_cny, Green_cny, Red_cny])
    # cv2.imshow('c', img_cny)
    edge_strength = np.uint8(edge_strength)
    edges = cv2.Canny(edge_strength, 30, 200)
    # cv2.imshow('e_s', edge_strength)
    # cv2.imshow('edge', edges)
    # cv2.waitKey(1)
    # cv2.destroyAllWindows()
    indices = np.nonzero(edges)
    coordinates = list(zip(indices[1], indices[0]))
    
    # vanishing_point = cal_vanishing_point(coordinates, local_orientation, width, height)
    # print('---'*10)
    # print(vanishing_point)
    # print('---'*10)
    vanishing_point = (152, 40)
    image = cv2.circle(frame, vanishing_point, radius=0, color=(0, 0, 255), thickness=10)
    # image = cv2.line(frame, (0, vanishing_point[1]), (width-1, vanishing_point[1]), color=(255, 0, 0), thickness=3)
    
    # number of rays 
    num_ray = 29
    j = 0
    ray_list = np.empty([num_ray, 2])
    for i in range(200, 341, 5):
        x = int(vanishing_point[0] + 200*math.cos(i*np.pi/180))
        y = int(vanishing_point[1] - 200*math.sin(i*np.pi/180))
        ray_list[j] = [x, y]
        j += 1
        image = cv2.line(frame, (vanishing_point[0], vanishing_point[1]), (x, y), color=(0,255,0), thickness=1)
    # spacing of a ray (used to calculate neighbor pixel)
    phi = np.pi/12
    N_r_list = []
    L = math.sqrt(height**2 + width**2)
    tau_e = 0.01
    norm = np.linalg.norm
    vanishing_point = np.asarray(vanishing_point)
    for i in range(1, len(ray_list)):
        N_r = []
        for y in range(vanishing_point[1], height):
            for x in range(width):
                p = np.asarray([x, y])
                dist = abs(np.cross(ray_list[i] - vanishing_point, p - vanishing_point) / norm(ray_list[i] - vanishing_point))
                if dist <= L*tau_e:
                    N_r.append((x, y))
        N_r_list.append(np.asarray(N_r))

    d0_list = []
    # angle of a ray
    theta_r = 240*np.pi/180
    for i in range(len(N_r_list)):
        d0 = 0
        for j in range(len(N_r_list[i])):
            x = N_r_list[i][j][0]
            y = N_r_list[i][j][1]
            if(y < 0 or y > height or x < 0 or x > width):
                continue
            d0 += abs(theta_r - local_orientation[y, x])
        d0_list.append(d0)
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