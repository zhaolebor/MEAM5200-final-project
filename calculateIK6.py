#from typing_extensions import Self
# calculateIK6.py
import numpy as np
from numpy import cos,sin,tan
from math import pi
from calculateFK import FK
import math
#from core.interfaces import ArmController

class IK:
    """
    This is the IK function (class) that solves the 6 DOF (joint 5 fixed) IK problem for Panda robot arm
    Author: Jiaqi Lian
    Note that this class only contains IK-related subroutines of the manipulation algorithm
    Please see "final.py" for the full object manipulation algorithm
    """
    #arm = ArmController()
    # offsets along x direction
    a1 = 0
    a2 = 0
    a3 = 0.0825
    a4 = 0.0825
    a5 = 0
    a6 = 0.088
    a7 = 0

    # offsets along z direction
    d1 = 0.333
    d2 = 0
    d3 = 0.316
    d4 = 0
    d5 = 0.384
    d6 = 0
    d7 = 0.210

    # offsets of joint angles
    phi7 = pi/4

    # custom variable for calculation
    l1 = np.sqrt(a3**2+d5**2)
    l2 = np.sqrt(a3**2+d3**2)
    alpha = np.arctan(a3/d5)
    beta = np.arctan(a3/d3)

    Psi6_init = pi/2-alpha      # Psi6 = Psi6_init + theta6
    Psi4_init = alpha + beta    # Psi4 = Psi4_init + theta4

    maximum_length = d3 + d5 + np.sqrt(a6**2+d7**2)


    # This variable is used to express an arbitrary joint angle
    Q0 = -0.12

    #joint limits dictionary

    joint_limits = [{'lower': -2.8973, 'upper': 2.8973},
                    {'lower': -1.7628, 'upper': 1.7628},
                    {'lower': -2.8973, 'upper': 2.8973},
                    {'lower': -3.0718, 'upper': -0.0698},
                    {'lower': -2.8973, 'upper': 2.8973},
                    {'lower': -0.0175, 'upper': 3.7525},
                    {'lower': -2.8973, 'upper': 2.8973}]


    def calculate_RandT_fromDH(self,ai,alphai,di,thetai):
        R = np.array([[cos(thetai),      -sin(thetai)*cos(alphai),      sin(thetai)*sin(alphai)],
                      [sin(thetai),      cos(thetai)*cos(alphai),       -cos(thetai)*sin(alphai)],
                      [0,                sin(alphai),                   cos(alphai)]])

        T = np.array([[cos(thetai),      -sin(thetai)*cos(alphai),      sin(thetai)*sin(alphai),      ai*cos(thetai)],
                      [sin(thetai),      cos(thetai)*cos(alphai),       -cos(thetai)*sin(alphai),     ai*sin(thetai)],
                      [0,                sin(alphai),                   cos(alphai),                  di],
                      [0,                0,                             0,                            1]])

        return R,T

    def findO_A2_fromO_72(self,O_72,theta7_val):
        #first calculate T_67 and find O2 in the frame of 6
        theta = theta7_val - self.phi7
        alpha = 0
        R_67,T_67 = self.calculate_RandT_fromDH(0,0,self.d7,theta)

        O_72 = np.append(O_72,[[1]],axis = 0)
        O_62 = T_67 @ O_72
        # Then, get O_A2 by applying offset
        O_62[0] = O_62[0] + self.a6
        O_A2 = O_62
        return O_A2


    def panda_ik(self, target):
        """
        Solves 6 DOF IK problem given physical target in x, y, z space
        Args:
            target: dictionary containing:
                'R': numpy array of the end effector pose relative to the robot base
                't': numpy array of the end effector position relative to the robot base

        Returns:
             q = nx7 numpy array of joints in radians (q5: joint 5 angle should be 0)
        """
        q = []

        # Student's code goes in between:
        joint_limits = self.joint_limits
        # calculate O_72
        # if in the boundary, pass to IK functions
        # if out of boundary, there will be no solution
        wrist_pos = self.kin_decouple(target)
        x_72,y_72,z_72 = wrist_pos[:,0]
        l_72 = np.sqrt(x_72**2+y_72**2+z_72**2)

        #print("Within the boundary?  ",l_72<=self.maximum_length)
        if l_72 <= self.maximum_length:
            joints_467 = self.ik_pos(wrist_pos)
            joints_123 = self.ik_orient(target['R'],joints_467)
            if len(joints_467) > 0:

                q_temp = np.hstack((joints_123,joints_467))
                if (target['R'][2][2]%1==0)and(target['t'][0]==0)and(target['t'][1]==0):
                    for i in range(len(q_temp)):
                        q_temp[i][0] = self.Q0
                        q_temp[i][5] = self.Q0
                joints_5 = np.zeros((len(joints_467),1))
                q_temp = np.insert(q_temp,[4],joints_5,axis = 1)

                #print("q_temp: \n",q_temp)
                q_pre = np.array([99,99,99,99,99,99,99])
                for i,qarr in enumerate(q_temp):
                    if not(np.array_equal(q_pre,qarr)):
                        if ((qarr[0] >= joint_limits[0]['lower']) and (qarr[0] <= joint_limits[0]['upper'])):
                            if ((qarr[1] >= joint_limits[1]['lower']) and (qarr[1] <= joint_limits[1]['upper'])):
                                if ((qarr[2] >= joint_limits[2]['lower']) and (qarr[2] <= joint_limits[2]['upper'])):
                                    q.append(qarr)
                    #else:
                    #    print("Delete one repeating q: \n",q_pre)
                    q_pre = qarr
            # else:
            #     print("No Joint_467 solutions!!!!")

        else:
            q = []

        if len(q) == 0:
            q = np.array([]).reshape(0, 7)
        else:
            q = np.asarray(q)
        #print("q before sorted:\n",q)

        #q = np.array([]).reshape(0, 7)
        # Student's code goes in between:

        ## DO NOT EDIT THIS PART
        # This will convert your joints output to the autograder format
        q = self.sort_joints(q)
        ## DO NOT EDIT THIS PART
        return q

    def kin_decouple(self, target):
        """
        Performs kinematic decoupling on the panda arm to find the position of wrist center
        Args:
            target: dictionary containing:
                'R': numpy array of the end effector pose relative to the robot base
                't': numpy array of the end effector position relative to the robot base

        Returns:
             wrist_pos = 3x1 numpy array of the position of the wrist center in frame 7

        R_07 ==> in the frame of 0
        """
        wrist_pos = []
        R_07 = target['R'].reshape(3,3)
        t_07 = target['t'].reshape(3,1)

        R_70 = R_07.T
        t_70 = -R_70 @ t_07

        # positions in the frame of end effector
        O_70 = t_70

        d1_dir = np.array([0,0,1]).reshape(3,1)
        wrist_pos = O_72 = t_70 + self.d1*R_70@d1_dir

        return wrist_pos

    def ik_pos(self, wrist_pos):
        """
        Solves IK position problem on the joint 4, 6, 7
        Args:
            wrist_pos: 3x1 numpy array of the position of the wrist center in frame 7

        Returns:
             joints_467 = nx3 numpy array of all joint angles of joint 4, 6, 7
        """
        joint_limits = self.joint_limits
        joints_467 = []
        alt_the4 = []
        alt_the6 = []
        alt_the7 = []
        xc,yc,zc = wrist_pos[:,0]

        # look for theta7
        c_angle = math.atan2(xc,yc)
        if c_angle < 0:
            c_angle += (2*pi)
        theta7_0 = c_angle-(5/4)*pi
        theta7_1 = theta7_0+pi
        theta7_2 = theta7_0-pi
        alt_the7_temp = np.array([theta7_0,theta7_1,theta7_2])

        for num_the7 in range(len(alt_the7_temp)):
            if ((alt_the7_temp[num_the7] >= joint_limits[6]['lower']) and (alt_the7_temp[num_the7] <= joint_limits[6]['upper'])):
                alt_the7.append(alt_the7_temp[num_the7])

        # look for corresponding theta6, and theta4
        for i,theta7 in enumerate(alt_the7):
            alt_the4 = [] #reset alt_the4
            O_A2 = self.findO_A2_fromO_72(wrist_pos,theta7)
            Ox = O_A2[0][0] #ox = x axis in O_A
            Oy = O_A2[2][0] #oy = z axis in O_A
            arccos_term = (Ox**2 + Oy**2 - self.l1**2 - self.l2**2)/(2*self.l1*self.l2)
            if abs(arccos_term) > 1:
                continue

            Psi4_0 = np.arccos(arccos_term)
            theta4_0 = Psi4_0 - self.Psi4_init

            # if Psi4 = 0, only one answer
            if Psi4_0 != 0:
                Psi4_1 = -Psi4_0
                theta4_1 = Psi4_1 - self.Psi4_init

                alt_the4_temp = [theta4_0,theta4_1]

            else:
                alt_the4_temp = [theta4_0]

            # Delete the solutions out of range
            for num_the4 in range(len(alt_the4_temp)):
                if ((alt_the4_temp[num_the4] >= joint_limits[3]['lower']) and (alt_the4_temp[num_the4] <= joint_limits[3]['upper'])):
                    alt_the4.append(alt_the4_temp[num_the4])

            # solve for theta6 with values from alt_the4&7
            for j,theta4 in enumerate(alt_the4):
                Psi6 = math.atan2(Oy,Ox)-math.atan2(self.l2*sin(theta4+self.Psi4_init),self.l1+self.l2*cos(theta4+self.Psi4_init))
                theta6 = Psi6 - self.Psi6_init
                if (theta6 < -0.0175):
                    theta6 += 2 * pi

                if ((theta6 >= joint_limits[5]['lower']) and (theta6 <= joint_limits[5]['upper'])):
                    joints_467.append([theta4,theta6,theta7])
                    joints_467.append([theta4,theta6,theta7])


        joints_467 = np.asarray(joints_467)

        for i, joints in enumerate(joints_467):
            R_23,T_23 = self.calculate_RandT_fromDH(self.a3,pi/2,self.d3,0)
            R_34,T_34 = self.calculate_RandT_fromDH(-self.a4,-pi/2,0,joints[0])
            R_45,T_45 = self.calculate_RandT_fromDH(0,pi/2,self.d5,0)
            R_56,T_56 = self.calculate_RandT_fromDH(self.a6,pi/2,0,joints[1])
            R_67,T_67 = self.calculate_RandT_fromDH(0,0,self.d7,joints[2]-self.phi7)

            T_27 = T_23 @ T_34 @ T_45 @ T_56 @ T_67
            R = T_27[:3,:3].reshape(3,3)
            t = T_27[:3,3:].reshape(3,1)
            Rinv = R.T
            tinv = -R.T@t
        return joints_467

    def ik_orient(self, R, joints_467):
        """
        Solves IK orientation problem on the joint 1, 2, 3
        Args:
            R: numpy array of the end effector pose relative to the robot base
            joints_467: nx3 numpy array of all joint angles of joint 4, 6, 7

        Returns:
            joints_123 = nx3 numpy array of all joint angles of joint 1, 2 ,3
        """
        joint_limits = self.joint_limits
        joints_123 = []
        alt_the1 = []
        alt_the2 = []
        alt_the3 = []
        R_07 = R
        R_70 = R_07.T

        for i in range(int(len(joints_467)/2)):
            theta4 = joints_467[i*2][0]
            theta5 = 0
            theta6 = joints_467[i*2][1]
            theta7 = joints_467[i*2][2]

            R_34,T_34 = self.calculate_RandT_fromDH(-self.a4,-pi/2,0,theta4)
            R_45,T_45 = self.calculate_RandT_fromDH(0,pi/2,self.d5,0)
            R_56,T_56 = self.calculate_RandT_fromDH(self.a6,pi/2,0,theta6)
            R_67,T_67 = self.calculate_RandT_fromDH(0,0,self.d7,theta7-self.phi7)

            T_37 = T_34 @ T_45 @ T_56 @ T_67
            R_37 = R_34@R_45@R_56@R_67
            R_37 = T_37[:3,:3].reshape(3,3)
            R_30 = R_37 @ R_70
            R_03 = R_30.T

            theta2_0 = np.arccos(round(R_03[2][1],5))

            theta2_1 = -theta2_0
            alt_the2 = np.array([theta2_0,theta2_1])

            # if theta 2 = 0, q1 and q3 is determined directly
            if (theta2_0 % pi) <= 1e-3:
                joints_123.append([self.Q0,theta2_0,self.Q0])
                joints_123.append([self.Q0,theta2_0,self.Q0])
                continue

            for j,theta2 in enumerate(alt_the2):
                cos_the1 = R_03[0][1]/(sin(theta2))
                sin_the1 = R_03[1][1]/(sin(theta2))

                theta1 = math.atan2(sin_the1,cos_the1)

                cos_the3 = R_03[2][0]/(-sin(theta2))
                sin_the3 = R_03[2][2]/(-sin(theta2))

                theta3 = math.atan2(sin_the3,cos_the3)

                joints_123.append([theta1,theta2,theta3])

        joints_123 = np.asarray(joints_123)


        return joints_123

    def sort_joints(self, q, col=0):
        """
        Sort the joint angle matrix by ascending order
        Args:
            q: nx7 joint angle matrix
        Returns:
            q_as = nx7 joint angle matrix in ascending order
        """

        if col != 7:
            q_as = q[q[:, col].argsort()]
            for i in range(q_as.shape[0]-1):
                if (q_as[i, col] < q_as[i+1, col]):
                    # do nothing
                    pass
                else:
                    for j in range(i+1, q_as.shape[0]):
                        if q_as[i, col] < q_as[j, col]:
                            idx = j
                            break
                        elif j == q_as.shape[0]-1:
                            idx = q_as.shape[0]

                    q_as_part = self.sort_joints(q_as[i:idx, :], col+1)
                    q_as[i:idx, :] = q_as_part
        else:
            q_as = q[q[:, -1].argsort()]
        return q_as

    def get_grasp_trans(self,target):
        '''
        This function inputs the target score block transformation matrix and returns the desired end effector grasping pose
        Input ==> target: target score block T, array(4,4)
        Output
        ==> R: R of end-effector while grasping, array(3,3)
        ==> t: t of end-effector while grasping, array(3,)
        '''
        R = target[:3,:3]
        if R[2,2] <= -0.9: # Z points down
        	adj = np.diag([1,1,1])

        elif R[2,2] >= 0.9: # Z points up
        	adj = np.diag([1,-1,-1])

        elif R[2,0] >= 0.9: # x points up
        	adj = np.array([[0,0,-1],
                			[0,1,0],
                			[1,0,0]])

        elif R[2,0] <= -0.9: # x points down
            adj = np.array([[0,0,1],
                    		[0,1,0],
                    		[-1,0,0]])

        elif R[2,1] >= 0.9: # y points up
        	adj = np.array([[1,0,0],
                			[0,0,-1],
                			[0,1,0]])

        elif R[2,1] <= -0.9: # y points down
            adj = np.array([[1,0,0],
                    		[0,0,1],
                    		[0,-1,0]])

        R = R @ adj
        t = target[:3,3].flatten()
        return R, t

    def find_grasp_IKconfig(self, arm, target, enforce=False):
        '''
        This function calculate configuration of grasping pose based on IK and find the cloest (optimized) config
        Input ==> target: target score block T, array(4,4)
        Output ==> q_opt: cloest/optimized IK configuration, array(7,)
        '''
        if not enforce:
            R,t = self.get_grasp_trans(target)
        else:
            R = target[:3,:3]
            t = target[:3,3].flatten()
        ee_pose = {
          'R':R,
          't':t
        }
        q_all = np.array([]).reshape(0,7)
        if not enforce:
            try_err = np.array([[0,-1,0],[1,0,0],[0,0,1]])

            for i in range(4):
                if i >= 1:
                    ee_pose['R'] = try_err @ ee_pose['R']
                q_ele = self.panda_ik(ee_pose)
                if len(q_ele) == 0:
                    continue
                else:
                    q_all = np.vstack((q_all,q_ele))
        else:
            q_ele = self.panda_ik(ee_pose)
            q_all = np.vstack((q_all,q_ele))

        if len(q_all) == 0:
            return np.array([])
        q_curr = arm.get_positions()
        mgn = [1,1,1,1,0,1,2]

        q_diff = q_all - q_curr
        q_err = mgn*q_diff
        q_err_norm = np.linalg.norm(q_err, axis = 1)
        min_idx = np.where(q_err_norm == q_err_norm.min())[0][0]
        q_opt = q_all[min_idx]
        #print("q_opt:",q_opt)
        return np.array(q_opt)

'''
if __name__ == "__main__":

    ik = IK()

    target = np.array([[6.686e-09,7.99532e-01,-6.006e-01,1],
    			[-9.829e-10,-6.0062e-01,-7.99532e-01,1],
    			[-1,5.93e-09,-3.23e-09,1],
    			[0,0,0,1]])
'''
