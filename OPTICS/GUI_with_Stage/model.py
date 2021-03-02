from configurations import *
import numpy as np
from math import *
import cv2
import imutils
import statistics
import time
import random


class Model(dict):
    init_value=0

    def __init__(self):
        self.init_value=1


    # def ik(self, a_matrix):
    #     a11=np.sum(a_matrix)
    #     return a11

    def mat_A(self, theta, d, a, alpha):
        # angles are degrees
        # d,a in meters
        theta_ = theta + 0.1
        theta__ = theta - 0.1
        theta = theta * pi / 180
        theta_ = theta_ * pi / 180
        theta__ = theta__ * pi / 180
        alpha = alpha * pi / 180

        A1 = np.array([[cos(theta), -sin(theta) * cos(alpha), sin(theta) * sin(alpha), a * cos(theta)],
                       [sin(theta), cos(theta) * cos(alpha), -cos(theta) * sin(alpha), a * sin(theta)],
                       [0, sin(alpha), cos(alpha), d],
                       [0, 0, 0, 1]])
        A1_ = np.array([[cos(theta_), -sin(theta_) * cos(alpha), sin(theta_) * sin(alpha), a * cos(theta_)],
                        [sin(theta_), cos(theta_) * cos(alpha), -cos(theta_) * sin(alpha), a * sin(theta_)],
                        [0, sin(alpha), cos(alpha), d],
                        [0, 0, 0, 1]])
        # A1__=np.array([[cos(theta__),-sin(theta__)*cos(alpha),sin(theta__)*sin(alpha),a*cos(theta__)],
        #             [sin(theta__),cos(theta__)*cos(alpha),-cos(theta__)*sin(alpha),a*sin(theta__)],
        #             [0,sin(alpha),cos(alpha),d],
        #             [0,0,0,1]])

        # return([A1, A1_,A1__])
        return ([A1, A1_])

    def mat_J(self, a1, a2, a3, a4, a5, a6):
        d = 0.1
        # [A1,A1_,A1__]=a1
        # [A2, A2_,A2__] = a2
        # [A3, A3_,A3__] = a3
        # [A4, A4_,A4__] = a4
        # [A5, A5_,A5__] = a5
        # [A6, A6_,A6__] = a6

        [A1, A1_] = a1
        [A2, A2_] = a2
        [A3, A3_] = a3
        [A4, A4_] = a4
        [A5, A5_] = a5
        [A6, A6_] = a6

        T0 = np.dot(np.dot(np.dot(np.dot(np.dot(A1, A2), A3), A4), A5), A6)
        T1 = np.dot(np.dot(np.dot(np.dot(np.dot(A1_, A2), A3), A4), A5), A6)
        T2 = np.dot(np.dot(np.dot(np.dot(np.dot(A1, A2_), A3), A4), A5), A6)
        T3 = np.dot(np.dot(np.dot(np.dot(np.dot(A1, A2), A3_), A4), A5), A6)
        T4 = np.dot(np.dot(np.dot(np.dot(np.dot(A1, A2), A3), A4_), A5), A6)
        T5 = np.dot(np.dot(np.dot(np.dot(np.dot(A1, A2), A3), A4), A5_), A6)
        T6 = np.dot(np.dot(np.dot(np.dot(np.dot(A1, A2), A3), A4), A5), A6_)

        # T1_ = np.dot(np.dot(np.dot(np.dot(np.dot(A1__, A2), A3), A4), A5), A6)
        # T2_ = np.dot(np.dot(np.dot(np.dot(np.dot(A1, A2__), A3), A4), A5), A6)
        # T3_ = np.dot(np.dot(np.dot(np.dot(np.dot(A1, A2), A3__), A4), A5), A6)
        # T4_ = np.dot(np.dot(np.dot(np.dot(np.dot(A1, A2), A3), A4__), A5), A6)
        # T5_ = np.dot(np.dot(np.dot(np.dot(np.dot(A1, A2), A3), A4), A5__), A6)
        # T6_ = np.dot(np.dot(np.dot(np.dot(np.dot(A1, A2), A3), A4), A5), A6__)

        # dT1d1=(T1-T1_)/2/d
        # dT2d2 = (T2 - T2_)/2/d
        # dT3d3 = (T3 - T3_)/2/d
        # dT4d4 = (T4 - T4_)/2/d
        # dT5d5 = (T5 - T5_)/2/d
        # dT6d6 = (T6 - T6_)/2/d
        dT1d1 = (T1 - T0) / d
        dT2d2 = (T2 - T0) / d
        dT3d3 = (T3 - T0) / d
        dT4d4 = (T4 - T0) / d
        dT5d5 = (T5 - T0) / d
        dT6d6 = (T6 - T0) / d

        J = np.array([[dT1d1[0, 0], dT2d2[0, 0], dT3d3[0, 0], dT4d4[0, 0], dT5d5[0, 0], dT6d6[0, 0]],
                      [dT1d1[0, 1], dT2d2[0, 1], dT3d3[0, 1], dT4d4[0, 1], dT5d5[0, 1], dT6d6[0, 1]],
                      [dT1d1[0, 2], dT2d2[0, 2], dT3d3[0, 2], dT4d4[0, 2], dT5d5[0, 2], dT6d6[0, 2]],
                      [dT1d1[1, 0], dT2d2[1, 0], dT3d3[1, 0], dT4d4[1, 0], dT5d5[1, 0], dT6d6[1, 0]],
                      [dT1d1[1, 1], dT2d2[1, 1], dT3d3[1, 1], dT4d4[1, 1], dT5d5[1, 1], dT6d6[1, 1]],
                      [dT1d1[1, 2], dT2d2[1, 2], dT3d3[1, 2], dT4d4[1, 2], dT5d5[1, 2], dT6d6[1, 2]],
                      [dT1d1[2, 0], dT2d2[2, 0], dT3d3[2, 0], dT4d4[2, 0], dT5d5[2, 0], dT6d6[2, 0]],
                      [dT1d1[2, 1], dT2d2[2, 1], dT3d3[2, 1], dT4d4[2, 1], dT5d5[2, 1], dT6d6[2, 1]],
                      [dT1d1[2, 2], dT2d2[2, 2], dT3d3[2, 2], dT4d4[2, 2], dT5d5[2, 2], dT6d6[2, 2]],
                      [dT1d1[0, 3], dT2d2[0, 3], dT3d3[0, 3], dT4d4[0, 3], dT5d5[0, 3], dT6d6[0, 3]],
                      [dT1d1[1, 3], dT2d2[1, 3], dT3d3[1, 3], dT4d4[1, 3], dT5d5[1, 3], dT6d6[1, 3]],
                      [dT1d1[2, 3], dT2d2[2, 3], dT3d3[2, 3], dT4d4[2, 3], dT5d5[2, 3], dT6d6[2, 3]]])

        return (J)

    def T_to_e(self,T):
        e = np.transpose(np.array([[T[0, 0], T[0, 1], T[0, 2], T[1, 0], T[1, 1], T[1, 2], T[2, 0], T[2, 1], T[2, 2],
                                    T[0, 3], T[1, 3], T[2, 3]]]))
        return (e)

    def FK(self,theta1, d1, a1, alpha1, theta2, d2, a2, alpha2, theta3, d3, a3, alpha3, theta4, d4, a4, alpha4, theta5, d5,
           a5, alpha5, theta6, d6, a6, alpha6):
        AA1 = self.mat_A(theta1, d1, a1, alpha1)
        AA2 = self.mat_A(theta2, d2, a2, alpha2)
        AA3 = self.mat_A(theta3, d3, a3, alpha3)
        AA4 = self.mat_A(theta4, d4, a4, alpha4)
        AA5 = self.mat_A(theta5, d5, a5, alpha5)
        AA6 = self.mat_A(theta6, d6, a6, alpha6)
        # [A1, A1_, A1__] = AA1
        # [A2, A2_, A2__] = AA2
        # [A3, A3_, A3__] = AA3
        # [A4, A4_, A4__] = AA4
        # [A5, A5_, A5__] = AA5
        # [A6, A6_, A6__] = AA6

        [A1, A1_] = AA1
        [A2, A2_] = AA2
        [A3, A3_] = AA3
        [A4, A4_] = AA4
        [A5, A5_] = AA5
        [A6, A6_] = AA6
        T0 = np.dot(np.dot(np.dot(np.dot(np.dot(A1, A2), A3), A4), A5), A6)
        return ([T0, AA1, AA2, AA3, AA4, AA5, AA6])

    def local_optimization(self, theta1, d1, a1, alpha1, theta2, d2, a2, alpha2, theta3, d3, a3, alpha3, theta4, d4, a4,
                           alpha4, theta5, d5, a5, alpha5, theta6, d6, a6, alpha6, e_ref):
        ang1 = [theta1 - 0.1, theta1, theta1 + 0.1]
        ang2 = [theta2 - 0.1, theta2, theta2 + 0.1]
        ang3 = [theta3 - 0.1, theta3, theta3 + 0.1]
        ang4 = [theta4 - 0.1, theta4, theta4 + 0.1]
        ang5 = [theta5 - 0.1, theta5, theta5 + 0.1]
        ang6 = [theta6 - 0.1, theta6, theta6 + 0.1]
        de_norm = np.empty(0)
        x = np.array([[100, 100, 100, 100, 100, 100], [100, 100, 100, 100, 100, 100]])

        for x1 in ang1:
            for x2 in ang2:
                for x3 in ang3:
                    for x4 in ang4:
                        for x5 in ang5:
                            for x6 in ang6:
                                [T, AA1, AA2, AA3, AA4, AA5, AA6] = self.FK(x1, d1, a1, alpha1, x2, d2, a2, alpha2, x3, d3,
                                                                       a3, alpha3, x4, d4, a4,
                                                                       alpha4, x5, d5, a5, alpha5, x6, d6, a6, alpha6)
                                de = e_ref - self.T_to_e(T)
                                de_norm = np.append(de_norm, np.linalg.norm(de))
                                x = np.append(x, [[x1, x2, x3, x4, x5, x6]], axis=0)

        min_index = np.argmin(de_norm)
        min_norm = min(de_norm)
        min_angles = np.array([x[min_index + 2]])
        return ([min_norm, min_angles])

    def color_plus_shape(self,cap):
        ret, frame = cap.read()
        # if frame is read correctly ret is True

        img = cv2.medianBlur(frame, 5)

        # Convert BGR to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # define range of blue color in HSV
        lower_blue = np.array([0, 10, 70])
        upper_blue = np.array([10, 255, 255])

        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(hsv, hsv, mask=mask)

        cimg0 = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        cimg1 = cv2.cvtColor(cimg0, cv2.COLOR_GRAY2BGR)

        circles = cv2.HoughCircles(cimg0, cv2.HOUGH_GRADIENT, 1, 100, param1=30, param2=30, minRadius=0, maxRadius=50)
        #
        if circles is None:
            pass
        else:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # draw the outer circle
                cv2.circle(res, (i[0], i[1]), i[2], (0, 255, 0), 10)
                # draw the center of the circle
                cv2.circle(res, (i[0], i[1]), 2, (0, 0, 255), 10)

        return (res)

    def frame_cal(self,cap, npz_filename):
        ret, frame = cap.read()
        # if frame is read correctly ret is True

        B = np.load(npz_filename)
        with np.load(npz_filename) as X:
            mtx, dist, _, _ = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]
        img1 = frame
        h, w = img1.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        # undistort
        dst = cv2.undistort(img1, mtx, dist, None, newcameramtx)

        # crop the image
        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]

        return dst

    def color_plus_shape_calibration(self,dst):
        # ret, frame = cap.read()
        # if frame is read correctly ret is True

        img = cv2.medianBlur(dst, 5)

        # Convert BGR to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # define range of blue color in HSV
        lower_blue = np.array([0, 70, 50])
        upper_blue = np.array([10, 255, 255])

        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(hsv, hsv, mask=mask)

        cimg0 = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        cimg1 = cv2.cvtColor(cimg0, cv2.COLOR_GRAY2BGR)

        circles = cv2.HoughCircles(cimg0, cv2.HOUGH_GRADIENT, 1, 20, param1=30, param2=30, minRadius=0, maxRadius=0)
        #
        if circles is None:
            pass
        else:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # draw the outer circle
                cv2.circle(res, (i[0], i[1]), i[2], (0, 255, 0), 10)
                # draw the center of the circle
                cv2.circle(res, (i[0], i[1]), 2, (0, 0, 255), 10)

        return (res)

    def ball_detection(self,dst0,dst1):
        img0 = cv2.GaussianBlur(dst0, (3, 3), 0)
        img1 = cv2.GaussianBlur(dst1, (3, 3), 0)

        # # Convert BGR to HSV
        hsv0 = cv2.cvtColor(img0, cv2.COLOR_BGR2HSV)
        mask0 = cv2.inRange(hsv0, lower_blue, upper_blue)
        mask0 = cv2.erode(mask0, None, iterations=1)
        mask0 = cv2.dilate(mask0, None, iterations=1)

        hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv1, lower_blue, upper_blue)
        mask1 = cv2.erode(mask1, None, iterations=1)
        mask1 = cv2.dilate(mask1, None, iterations=1)

        #
        cnts0 = cv2.findContours(mask0.copy(), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
        cnts0 = imutils.grab_contours(cnts0)
        center0 = None

        cnts1 = cv2.findContours(mask1.copy(), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
        cnts1 = imutils.grab_contours(cnts1)
        center1 = None

        # only proceed if at least one contour was found
        if len(cnts0) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c0 = max(cnts0, key=cv2.contourArea)
            ((x0, y0), radius0) = cv2.minEnclosingCircle(c0)
            M0 = cv2.moments(c0)
            center0 = (int(M0["m10"] / M0["m00"]), int(M0["m01"] / M0["m00"]))

            # only proceed if the radius meets a minimum size
            if radius0 > -1:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(dst0, (int(x0), int(y0)), int(radius0),
                           (0, 255, 255), 2)
                cv2.circle(dst0, center0, 5, (0, 0, 255), -1)
            # cv2.circle(dst0, center0, 5, (0, 0, 255), -1)

        # only proceed if at least one contour was found
        if len(cnts1) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c1 = max(cnts1, key=cv2.contourArea)
            ((x1, y1), radius1) = cv2.minEnclosingCircle(c1)
            M1 = cv2.moments(c1)
            center1 = (int(M1["m10"] / M1["m00"]), int(M1["m01"] / M1["m00"]))

            # only proceed if the radius meets a minimum size
            if radius1 > -1:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(dst1, (int(x1), int(y1)), int(radius1),
                           (0, 255, 255), 2)
                cv2.circle(dst1, center1, 5, (0, 0, 255), -1)
            # cv2.circle(dst1, center1, 5, (0, 0, 255), -1)
        return ([dst0,dst1])

    def xyz_estimation(self, dst0, dst1):
        x_better=0.0
        y_better=0.0
        z_better=0.0
        img0 = cv2.GaussianBlur(dst0, (3, 3), 0)
        img1 = cv2.GaussianBlur(dst1, (3, 3), 0)

        # # Convert BGR to HSV
        hsv0 = cv2.cvtColor(img0, cv2.COLOR_BGR2HSV)
        mask0 = cv2.inRange(hsv0, lower_blue, upper_blue)
        mask0 = cv2.erode(mask0, None, iterations=1)
        mask0 = cv2.dilate(mask0, None, iterations=1)

        hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv1, lower_blue, upper_blue)
        mask1 = cv2.erode(mask1, None, iterations=1)
        mask1 = cv2.dilate(mask1, None, iterations=1)

        #
        cnts0 = cv2.findContours(mask0.copy(), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
        cnts0 = imutils.grab_contours(cnts0)
        center0 = None

        cnts1 = cv2.findContours(mask1.copy(), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
        cnts1 = imutils.grab_contours(cnts1)
        center1 = None

        # only proceed if at least one contour was found
        if len(cnts0) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c0 = max(cnts0, key=cv2.contourArea)
            ((x0, y0), radius0) = cv2.minEnclosingCircle(c0)
            M0 = cv2.moments(c0)
            center0 = (int(M0["m10"] / M0["m00"]), int(M0["m01"] / M0["m00"]))

            # only proceed if the radius meets a minimum size
            if radius0 > 0:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(dst0, (int(x0), int(y0)), int(radius0),
                           (0, 255, 255), 2)
                cv2.circle(dst0, center0, 5, (0, 0, 255), -1)

        # only proceed if at least one contour was found
        if len(cnts1) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c1 = max(cnts1, key=cv2.contourArea)
            ((x1, y1), radius1) = cv2.minEnclosingCircle(c1)
            M1 = cv2.moments(c1)
            center1 = (int(M1["m10"] / M1["m00"]), int(M1["m01"] / M1["m00"]))

            # only proceed if the radius meets a minimum size
            if radius1 > 0:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(dst1, (int(x1), int(y1)), int(radius1),
                           (0, 255, 255), 2)
                cv2.circle(dst1, center1, 5, (0, 0, 255), -1)
            # xyz estimation
        if len(cnts0) > 0 and len(cnts1) > 0:
            # cam properties & z estimate
            width_pixel1 = img0.shape[1]
            height_pixel1 = img0.shape[0]
            width_pixel2 = img1.shape[1]
            height_pixel2 = img1.shape[0]
            x1 = center0[0]
            x2 = center1[0]
            y1 = center0[1]
            y2 = center1[1]

            #
            xl = (x1 - width_pixel1 / 2) / width_pixel1 * width_sensor
            xr = (x2 - width_pixel2 / 2) / width_pixel2 * width_sensor
            yl = (y1 - height_pixel1 / 2) / height_pixel1 * height_sensor
            yr = (y2 - height_pixel2 / 2) / height_pixel2 * height_sensor

            d = abs(xl - xr)

            z_adjust_index = 0.8245

            z = f * b / d * z_adjust_index  # in mm

            z_compensation_para = np.array(
                [x1 - width_pixel1 / 2, y1 - height_pixel1 / 2, x2 - width_pixel2 / 2, y2 - height_pixel2 / 2,
                 z / 25.4, 1])  # z/25.4 is because it's fitted in inch

            z_ratio_comp = np.dot(z_compensation, z_compensation_para)
            z_better = z / (z_ratio_comp)  # in mm

            x_adjust_index = 2
            y_adjust_index = 4.5

            x10 = xl * z_better / f * x_adjust_index  # from left camera # in mm
            x20 = b + xr * z_better / f * x_adjust_index  # from right camera # in mm
            x = statistics.mean([x10, x20])

            x_compensation_para = np.array(
                [x1 - width_pixel1 / 2, y1 - height_pixel1 / 2, x1 - width_pixel2 / 2, y1 - height_pixel2 / 2,
                 x, z_better, 1])
            x_error_comp = np.dot(x_compensation, x_compensation_para)
            x_better = x - x_error_comp

            y10 = -yl * z_better / f * y_adjust_index  # from left cameqra # in mm
            y20 = -yr * z_better / f * y_adjust_index  # from right camera # in mm
            y = statistics.mean([y10, y20])

            y_compensation_para = np.array(
                [x1 - width_pixel1 / 2, y1 - height_pixel1 / 2, x2 - width_pixel2 / 2, y2 - height_pixel2 / 2,
                 y, z_better, 1])
            y_error_comp = np.dot(y_compensation, y_compensation_para)
            y_better = y - y_error_comp
        # return([x_better/10+95,y_better/10+108,z_better/25.4])
        return ([x_better, y_better, z_better]) # unit in mm, relative to the coordinate of the center of cameras

            # # print(img0.shape[0],img0.shape[1],img1.shape[0],img1.shape[1])
            # #
            # # print("x=", x/25.4, "y=", y/25.4, "z=", z/25.4, "z better=", z_better/25.4,"z ratio=",z_ratio_comp)
            # # print(y, y_error_comp)
            # print("--- %s seconds ---" % (time.time() - start_time))
            # print("x_better=", x_better / 10 + 95, "y_better=", y_better / 10 + 108, "z better=",
            #       z_better / 25.4)  # coordinate relative to the initial measuring tape steup, x_better, y_better _z_better are
            # # relative to the center point of two cameras

    def decision_tree_landing_estimation(self, x_shoot, y_shoot, timestamp_shoot):
        rela_x0_cm=x_shoot[0]/10
        rela_x1_cm=x_shoot[1]/10
        rela_y0_cm = y_shoot[0] / 10
        rela_y1_cm = y_shoot[1] / 10

        vx_cm = ((rela_x1_cm - rela_x0_cm) / (timestamp_shoot[1] - timestamp_shoot[0]))
        vy_cm = ((rela_y1_cm - rela_y0_cm ) / (timestamp_shoot[1] - timestamp_shoot[0]))

        poolx0 = [-1, 1]
        poolx1 = [-1, -1, -1, -1, -1, -1, -1, 1]
        poolx2 = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        if vx_cm <= -7.5:
            result_x=0
        else:
            if vx_cm <= 20.5:
                result_x = -1
            else:
                if rela_x0_cm <= -19.5:
                    result_x = 0
                else:
                    result_x = 1

        pooly0 = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
                  - 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

        if rela_y0_cm <= -3.5:
            result_y = -1
        else:
            if vy_cm <= -170.5:
                result_y = 1
            else:
                result_y = 0

        output = 0

        if result_x == 1:
            if result_y == 1:
                output = 1
            elif result_y == 0:
                output = 2
            elif result_y == -1:
                output = 3
        elif result_x == 0:
            if result_y == 1:
                output = 4
            elif result_y == 0:
                output = 5
            elif result_y == -1:
                output = 6
        elif result_x == -1:
            if result_y == 1:
                output = 7
            elif result_y == 0:
                output = 8
            elif result_y == -1:
                output = 9

        return (output)
