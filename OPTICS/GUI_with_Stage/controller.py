from configurations import *
import model
import numpy as np
from math import *

class Controller():

    def __init__(self):
        self.init_model()

    def init_model(self):
        self.model = model.Model()

    def ik_calculation(self, a_matrix):
        vector_theta_0 = np.transpose([[theta1, theta2, theta3, theta4, theta5, theta6]])
        [T0, AA1, AA2, AA3, AA4, AA5, AA6] = self.model.FK(theta1, d1, a1, alpha1, theta2, d2, a2, alpha2, theta3, d3, a3,
                                                   alpha3, theta4, d4, a4, alpha4, theta5, d5, a5, alpha5, theta6, d6,
                                                   a6, alpha6)
        e0 = self.model.T_to_e(T0)

        e_ref = np.transpose([[a_matrix[0], a_matrix[1], a_matrix[2],
                               a_matrix[3], a_matrix[4], a_matrix[5],
                               a_matrix[6], a_matrix[7], a_matrix[8],
                               a_matrix[9], a_matrix[10], a_matrix[11],
                               ]])
        e = e0.copy()
        vector_theta = vector_theta_0.copy().astype(float)

        error_i = 0
        error_last = 0
        error = e_ref - e0
        de = e_ref - e0
        e9 = np.empty(0)
        e10 = np.empty(0)
        e11 = np.empty(0)
        angle1 = np.empty(0)
        angle2 = np.empty(0)
        angle3 = np.empty(0)
        angle4 = np.empty(0)
        angle5 = np.empty(0)
        angle6 = np.empty(0)
        de_norm = np.empty(0)

        for i in range(1, 2001):
            # IK calculation
            J = self.model.mat_J(AA1, AA2, AA3, AA4, AA5, AA6)
            # J_psuedo=np.dot(np.linalg.inv(np.dot(np.transpose(J),J)),np.transpose(J))
            J_ = np.transpose(J)

            error_last = error.copy()
            error = e_ref - e
            error_i += e_ref - e
            error_d = error - error_last
            dtheta = np.dot(J_, (error * kp + error_i * ki + error_d * kd))
            # dtheta = np.dot(J_*1500, de)
            vector_theta += dtheta

            ## angle to e for feedback
            # AA1 = fn.mat_A(vector_theta[0], d1, a1, alpha1)
            # AA2 = fn.mat_A(vector_theta[1], d2, a2, alpha2)
            # AA3 = fn.mat_A(vector_theta[2], d3, a3, alpha3)
            # AA4 = fn.mat_A(vector_theta[3], d4, a4, alpha4)
            # AA5 = fn.mat_A(vector_theta[4], d5, a5, alpha5)
            # AA6 = fn.mat_A(vector_theta[5], d6, a6, alpha6)
            #
            # [A1, A1_, A1__] = AA1
            # [A2, A2_, A2__] = AA2
            # [A3, A3_, A3__] = AA3
            # [A4, A4_, A4__] = AA4
            # [A5, A5_, A5__] = AA5
            # [A6, A6_, A6__] = AA6
            # T = np.dot(np.dot(np.dot(np.dot(np.dot(A1, A2), A3), A4), A5), A6)
            [T, AA1, AA2, AA3, AA4, AA5, AA6] = self.model.FK(vector_theta[0], d1, a1, alpha1, vector_theta[1], d2, a2, alpha2,
                                                      vector_theta[2], d3, a3, alpha3, vector_theta[3], d4, a4, alpha4,
                                                      vector_theta[4],
                                                      d5, a5, alpha5, vector_theta[5], d6, a6, alpha6)
            e = self.model.T_to_e(T)
            de = e_ref - e

            de_norm = np.append(de_norm, np.linalg.norm(de))
            e9 = np.append(e9, e[9])
            e10 = np.append(e10, e[10])
            e11 = np.append(e11, e[11])
            angle1 = np.append(angle1, vector_theta[0])
            angle2 = np.append(angle2, vector_theta[1])
            angle3 = np.append(angle3, vector_theta[2])
            angle4 = np.append(angle4, vector_theta[3])
            angle5 = np.append(angle5, vector_theta[4])
            angle6 = np.append(angle6, vector_theta[5])

        # locally search IK best results
        min_index = np.argmin(de_norm)
        min_angles = np.array(
            [angle1[min_index], angle2[min_index], angle3[min_index], angle4[min_index], angle5[min_index],
             angle6[min_index], ])

        # [opti_min_norm, opti_min_angles] = self.model.local_optimization(min_angles[0], d1, a1, alpha1, min_angles[1], d2, a2,
        #                                                          alpha2, min_angles[2], d3, a3, alpha3, min_angles[3],
        #                                                          d4, a4, alpha4, min_angles[4], d5, a5, alpha5,
        #                                                          min_angles[5], d6, a6, alpha6, e_ref)

        return self.model.local_optimization(min_angles[0], d1, a1, alpha1, min_angles[1], d2, a2,
                                                                 alpha2, min_angles[2], d3, a3, alpha3, min_angles[3],
                                                                 d4, a4, alpha4, min_angles[4], d5, a5, alpha5,
                                                                 min_angles[5], d6, a6, alpha6, e_ref)
    def ball_detection(self,cap0,cap1,npz_filename0,npz_filename1):
        dst0=self.model.frame_cal(cap0,npz_filename0)
        dst1 = self.model.frame_cal(cap1, npz_filename1)
        # dst=self.model.ball_detection(dst)
        return self.model.ball_detection(dst0,dst1)

    def xyz_estimation(self,cap0,cap1,npz_filename0,npz_filename1):
        dst0=self.model.frame_cal(cap0,npz_filename0)
        dst1 = self.model.frame_cal(cap1, npz_filename1)
        return self.model.xyz_estimation(dst0, dst1)

    def xy_landing_estimation(self, x_shoot, y_shoot, timestamp_shoot):
        return self.model.decision_tree_landing_estimation(x_shoot, y_shoot, timestamp_shoot)

    # def cam_calibration(self,cap, npz_filename):
    #     return self.model.frame_cal(cap, npz_filename)
    #
    # def calibration_frame_mask(self, dst):
    #     return self.model.color_plus_shape_calibration(dst)