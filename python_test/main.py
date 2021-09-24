#!/usr/bin/env python

import os
# import sys
# sys.path.append('/home/bernard/.local/lib/python3.7/site-packages')
import cv2
import numpy as np
import scipy.ndimage
import math
import copy
# from matplotlib import pyplot as plt
global width, height
# diagonal length of img
global L
# spacing of a ray (used to calculate neighbor pixel)
phi = np.pi/12
# the threshold for L needed for N_r 
tau_e = 0.01
# threshold on the orientation similarity between p and l_vp
tau_o = np.pi/12 # np.pi / 36
# angle of a ray
theta_r = 200*np.pi/180
# number of rays 
num_ray = 29
# angle to the next ray
ray_angle_diff = 5*np.pi/180
# number of bins in the color histogram
bins = 16

a = 0.9
b = 2.3
alpha = 0.9
beta = 2.3
sigma = 0.3437
phi_avg = 1.597


def image_grad(img):

    vertical_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    horizontal_filter = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    img = cv2.GaussianBlur(img, (5, 5), 0)
    # x_grad = scipy.ndimage.correlate(img, horizontal_filter)
    # y_grad = scipy.ndimage.correlate(img, vertical_filter)
    x_grad = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    y_grad = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

    # cv2.imshow('x', x_grad)
    # cv2.imshow('y', y_grad)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
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

def calc_vanishing_point(edge_coordinates, local_orientation):
    """
    Calculation for the vanishing point
    """
    global width, height, L
    
    vp_score_max = 0
    vanishing_point_list = list()
    delta_vp_list = list()
    vanishing_point = None
    from tqdm import tqdm
    # for y_ind in tqdm(range(height)):
    #     for x_ind in range(width):
    for y_ind, x_ind in tqdm(zip(range(height), range(width))):
        vp_score_sum = 0
        for i, j in edge_coordinates:
            if y_ind >= j: 
                continue
            l_vp = math.atan2(y_ind - j, x_ind - i) 
            delta_vp = abs(local_orientation[y_ind, x_ind] -  l_vp)
            # delta_vp_list.append(math.degrees(delta_vp))
            # if math.degrees(delta_vp) <= tau_o:
            if delta_vp <= tau_o:
                mu_vp = math.sqrt((y_ind - j)**2 + (x_ind - i)**2) / L
                vp_score = math.exp(-delta_vp * mu_vp)
            else:
                vp_score = 0
            vp_score_sum += vp_score
        # vp_score_max = max(vp_score_sum, vp_score_max)
        if vp_score_max < vp_score_sum:
            vp_score_max = vp_score_sum
            # vanishing_point = (y_ind, x_ind)
            vanishing_point = (x_ind, y_ind)
        else:
            vp_score_max = vp_score_max
            
        print('---'*10)
        print("Score: ", vp_score_max, " -> ", vanishing_point)
        print('---'*10)
        # vanishing_point_list.empty()
        # delta_vp_list.empty()
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
        d_o_list.append(d_o/len(N_r_list[i]))

    return R_list, N_r_list, d_o_list

def calc_d_c(R_list):
    """
    Calculation for d_c (color difference)
    """
    color_seg_img = cv2.imread("0_pred.png")
    color_seg_img = cv2.resize(color_seg_img, (width, height), interpolation = cv2.INTER_AREA)
    d_c_list = []
    for i in range(len(R_list)):
        d_c = []
        for j in range(len(R_list[i])):
            c_plus = np.zeros(3)
            c_minus = np.zeros(3)
            for plus, minus in zip(R_list[i][0], R_list[i][1]):
                c_plus += color_seg_img[plus[1], plus[0]]
                c_minus += color_seg_img[minus[1], minus[0]]
                
        c_plus /= len(R_list[i][0])    
        c_minus /= len(R_list[i][1])
        # d_c.append([c_plus, c_minus])  
        d_c = np.sqrt((c_plus[0] - c_minus[0])**2 + (c_plus[1] - c_minus[1])**2 + (c_plus[2] - c_minus[2])**2) / max(np.sqrt(np.sum(np.square(c_plus))), np.sqrt(np.sum(np.square(c_minus))))
        d_c_list.append(d_c)
    
    return d_c_list

def calc_u_ij(vanishing_point, ray_i, ray_j):
    '''
    Calculate color uniformity between ray "i" and ray "j" 
    '''
    ray_i = np.short(ray_i)
    ray_j = np.short(ray_j)
    left_x = [vanishing_point[0], ray_i[0]]
    left_y = [vanishing_point[1], ray_i[1]]
    right_x = [vanishing_point[0], ray_j[0]]
    right_y = [vanishing_point[1], ray_j[1]]
    left_fit = np.polyfit(left_x, left_y, 1)
    right_fit = np.polyfit(right_x, right_y, 1)

    ray_area_x_min = min(min(left_x), min(right_x))
    ray_area_x_max = max(max(left_x), max(right_x))
    plot_x = np.linspace(ray_area_x_min, ray_area_x_max, ray_area_x_max - ray_area_x_min)
    
    left_fit_y = left_fit[0]*plot_x + left_fit[1]
    right_fit_y = right_fit[0]*plot_x + right_fit[1]
    left_pts = np.array([(np.vstack([plot_x, left_fit_y])).T])
    right_pts = np.array([np.flipud((np.vstack([plot_x, right_fit_y])).T)])

    left_pts = np.array(left_pts, dtype='int32')
    right_pts = np.array(right_pts, dtype='int32')
    pts = np.hstack((left_pts, right_pts))
    # # Display ray area
    # cv2.fillPoly(img, pts, (0, 255, 0))
    # cv2.polylines(img, left_pts, isClosed=False, color=(255, 0, 0))
    # cv2.polylines(img, right_pts, isClosed=False, color=(255, 0, 0))
    color_seg_img = cv2.imread("frame004429.jpg_pred.png")
    color_seg_img = cv2.resize(color_seg_img, (width, height), interpolation = cv2.INTER_AREA)
    color_seg_img_copy = copy.deepcopy(color_seg_img)
    mask = np.zeros(color_seg_img_copy.shape[:2], dtype="uint8")
    mask = cv2.fillPoly(mask, pts, 1) # 255
    masked = cv2.bitwise_and(color_seg_img_copy, color_seg_img_copy, mask=mask)
    # cv2.imshow('mask', masked)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    chans = cv2.split(color_seg_img)
    colors = ("b", "g", "r")
    uniformity = 0
    for i, col in enumerate(colors):
        hist = cv2.calcHist([color_seg_img_copy], [i], mask, [bins], [0, 256])
        hist /= hist.sum()
        uniformity += np.sum(hist**2)
        # plt.plot(hist, color=col)
        # plt.xlim([0,256])
    # plt.show()
    return uniformity

def lane_score(i, j, d_o_list, d_c_list, uniformity, phi_list):
    angle = abs(phi_list[i]-phi_list[j])
    f1_i = np.exp(-d_o_list[i]/np.pi)
    f1_j = np.exp(-d_o_list[j]/np.pi)
    f2_i = 1/(1+a*np.exp(-b*d_c_list[i]))
    f2_j = 1/(1+a*np.exp(-b*d_c_list[j])) 
    f3 = 1/(1+alpha*np.exp(-beta*uniformity))
    f4 = 1/(sigma*math.sqrt(2*math.pi)) * math.exp(-(angle-phi_avg)**2/(2*(sigma)**2))
    score = f1_i*f1_j*f2_i*f2_j*f3*f4
    return score

def main():
    global width, height, L
    # file_dir = "/home/bernard/Akrobotix/code/picture/"
    # data = [i for i in os.listdir(file_dir) if i.endswith('_pred.png')]
    # print(len(data))
    # test_pic = cv2.imread("/home/bernard/Akrobotix/code/picture/0.png")
    # img = cv2.imread("/home/bernard/Akrobotix/code/0.png")
    img = cv2.imread("frame002730.jpg.png")
    # img = cv2.imread("frame002341.jpg.png")
    # img = cv2.imread("0.png")
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
    edges = cv2.Canny(edge_strength, 0, 200)
    # cv2.imshow('e_s', edge_strength)
    # cv2.imshow('edge', edges)
    # # cv2.imwrite('edge.png', edges)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    indices = np.nonzero(edges)
    coordinates = list(zip(indices[1], indices[0]))
    
    vanishing_point = calc_vanishing_point(coordinates, local_orientation)
    image = cv2.circle(img, vanishing_point, radius=0, color=(0, 0, 255), thickness=10)
    cv2.imshow('img', image)
    # cv2.imwrite('vanishing_point.png', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print('---'*10)
    print(vanishing_point)
    print('---'*10)
    # image = cv2.line(img, (0, vanishing_point[1]), (width-1, vanishing_point[1]), color=(255, 0, 0), thickness=3)
    
    j = 0
    ray_list = np.empty([num_ray, 2])
    phi_list = list()
    for i in range(200, 341, 5):
        phi_list.append(i*np.pi/180)
        x = int(vanishing_point[0] + 200*math.cos(i*np.pi/180))
        y = int(vanishing_point[1] - 200*math.sin(i*np.pi/180))
        if x < 0: 
            x = 0
        elif x > width: 
            x = width
        elif y > height: 
            y = height

        ray_list[j] = [x, y]
        j += 1
        # image = cv2.line(img, (vanishing_point[0], vanishing_point[1]), (x, y), color=(0,255,0), thickness=1)
    
    R_list, N_r_list, d_o_list = calc_d_o(img, local_orientation, vanishing_point, ray_list)

    d_c_list = calc_d_c(R_list)

    # color_uniformity = dict()
    score_dict = dict()
    for i in range(len(ray_list)-1):
        # color_uniformity[i] = list()
        score_dict[i] = (0,0)
        u_max = 0
        for j in range(i+1, len(ray_list)):
            uniformity = calc_u_ij(vanishing_point, ray_list[i], ray_list[j])
            # color_uniformity[i].append(uniformity)
            score = lane_score(i, j, d_o_list, d_c_list, uniformity, phi_list)
            if score > score_dict[i][1]:
                score_dict[i] = (j, score)
    val = 0
    for key in score_dict:
        if score_dict[key][1] > val:
            val = score_dict[key][1]
            ans = (key, score_dict[key][0])
    print(ans)
    print("done")
    img = cv2.line(img, (vanishing_point[0], vanishing_point[1]), (int(ray_list[ans[0]][0]), int(ray_list[ans[0]][1])), color=(0,255,0), thickness=1)
    img = cv2.line(img, (vanishing_point[0], vanishing_point[1]), (int(ray_list[ans[1]][0]), int(ray_list[ans[1]][1])), color=(0,255,0), thickness=1)
    cv2.imshow('img', img)
    cv2.imwrite('ray.jpg', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("done")
    # cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()