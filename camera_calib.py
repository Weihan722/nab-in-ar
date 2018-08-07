#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 15:05:39 2018

@author: jiweihan
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob


def grid_search_focal_length(points3d, points2d, h, w, same_f=False, fx_step=20):
    """ Finds the focal length that minimizes the reprojection error between a set of 3D points and its corresponding
    2D location. It searchs over the predefined interval 0.5, 1.5 x width
    :param points3d: set of 3D points
    :param points2d: its corresponding 2d points in the image
    :param h: height of the image
    :param w: width of the image
    :param same_f: if focal length in x and y direction are the same
    :param fx_step: step size of focal length search
    :return: the best focal length
    """
    best_score = 10e10
    best_fx, best_fy = -1, -1
    min_fx, max_fx = w/2., 10.*w
    min_fy, max_fy, fy_step = h / 2., 10. * h, fx_step
    if same_f:
        min_fy, max_fy, fy_step = 0, 1, 1

    for i in np.arange(min_fx, max_fx, fx_step):
        fx = i

        for j in np.arange(min_fy, max_fy, fy_step):
            if same_f:
                fy = fx
            else:
                fy = j

            A = intrinsic_matrix_from_focal_length(fx, fy, h, w)
            _, rvec, tvec = cv2.solvePnP(points3d, points2d, A, None)
            reproj, _ = cv2.projectPoints(points3d, rvec, tvec, A, None)
            reproj = np.squeeze(reproj)

            score = np.sum(np.linalg.norm(points2d - reproj, axis=1))

            # print(fx, fy, score)
            if score < best_score:
                best_score = score
                best_fx = fx
                best_fy = fy

    return best_fx, best_fy

def intrinsic_matrix_from_focal_length(fx, fy, h, w):
    return np.array([[fx, 0, w/2.], [0, fy, h/2.], [0, 0, 1]])

camera_pos = []

for filename in glob.glob('data//*.png'):
    field_img = cv2.imread('court_image.png',0)
    img = cv2.imread(filename, 0)
    
    
    
    h, w = img.shape[0:2]
    h2, w2 = field_img.shape[0:2]
    W, H = 28.65, 15.24
    
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img)
    ax[1].imshow(field_img)
    
    ax[0].axis('off')
    ax[1].axis('off')
    
    points2d = []
    points3d = []
    
    def onclick(event):
        print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f\n' %
              (event.button, event.x, event.y, event.xdata, event.ydata))
        x, y = event.xdata, event.ydata
        if event.inaxes.axes.get_position().x0 < 0.5:
            ax[0].plot(x, y, 'r.', ms=10)
            points2d.append([x, y])
        else:
            ax[1].plot(x, y, 'b+', ms=10)
            points3d.append([x, 0, y])
        plt.show()
    
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    points2d = np.array(points2d)
    points3d = np.array(points3d)
    
    
    points3d[:, 0] = ((points3d[:, 0] - w2 / 2.) / w2) * W
    points3d[:, 2] = ((points3d[:, 2] - h2 / 2.) / h2) * H
    
    #print points2d
    #print points3d
    
    points_3d_cv = points3d[:, np.newaxis, :].astype(np.float32)
    points_2d_cv = points2d[:, np.newaxis, :].astype(np.float32)
    
    fx, fy = grid_search_focal_length(points3d, points2d, h, w, same_f=True)
    A = intrinsic_matrix_from_focal_length(fx, fy, h, w)
    
    _, rvec, tvec, _ = cv2.solvePnPRansac(points_3d_cv, points_2d_cv, A, None)
    rvec, tvec = np.squeeze(rvec), np.squeeze(tvec)
    R, _ = cv2.Rodrigues(rvec)
    T = np.array([tvec]).T
    
    print A
    print R
    print T
    camera_pos.append((A, R, T))


