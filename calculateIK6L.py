import random

import numpy as np
from math import pi
import math
from numpy import random
from calculateFKL import FK

"""
Author: Lei Sun (ID: 61330435)
"""

class IK:
    """
    This is the standard IK function (class) that solves the 6 DOF (joint 5 fixed) IK problem for Panda robot arm
    Author: Lei Sun
    Note that this class only contains IK-related subroutines of the manipulation algorithm
    Please see "final.py" for the full object manipulation algorithm
    """
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
    
    # This variable is used to express an arbitrary joint angle 
    Q0 = 0.123



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

        T_07=self.get_transformation(target)

        wrist_pos=self.kin_decouple(target)

        joints_467 = self.ik_pos(wrist_pos)

        q_sol_4567=[]
        for i in range(len(joints_467)):
            q_sol_4567.append(list(joints_467[i]))


        q_sol=[]
        count_=0

        for num_sol in range(len(q_sol_4567)):
        
            q4=q_sol_4567[num_sol][0]
            q5=q_sol_4567[num_sol][1]
            q6=q_sol_4567[num_sol][2]
            q7=q_sol_4567[num_sol][3]

            T_34_=np.array([[np.cos(q4), -np.sin(q4)*np.cos(-pi/2), np.sin(q4)*np.sin(-pi/2), -self.a4*np.cos(q4)],
              [np.sin(q4), np.cos(q4)*np.cos(-pi/2), -np.cos(q4)*np.sin(-pi/2), -self.a4*np.sin(q4)],
              [0, np.sin(-pi/2), np.cos(-pi/2), 0],
              [0,0,0,1] ])

            T_45_=np.array([[np.cos(q5), -np.sin(q5)*np.cos(pi/2), np.sin(q5)*np.sin(pi/2), -0*np.cos(q5)],
              [np.sin(q5), np.cos(q5)*np.cos(pi/2), -np.cos(q5)*np.sin(pi/2), -0*np.sin(q5)],
              [0, np.sin(pi/2), np.cos(pi/2), self.d5],
              [0,0,0,1]])
        
            T_56_=np.array([[np.cos(q6), -np.sin(q6)*np.cos(pi/2), np.sin(q6)*np.sin(pi/2), self.a6*np.cos(q6)],
              [np.sin(q6), np.cos(q6)*np.cos(pi/2), -np.cos(q6)*np.sin(pi/2), self.a6*np.sin(q6)],
              [0, np.sin(pi/2), np.cos(pi/2), 0],
              [0,0,0,1] ])
        
            T_67_=np.array([[np.cos(q7-pi/4), -np.sin(q7-pi/4)*np.cos(0), 0, 0],
              [np.sin(q7-pi/4), np.cos(q7-pi/4)*np.cos(0), 0, 0],
              [0, np.sin(0), np.cos(0), self.d7],
              [0,0,0,1] ])

            T_37_=T_34_@T_45_@T_56_@T_67_
    
            T_03_=T_07@np.linalg.inv(T_37_)

            if np.abs(T_03_[2,1]-1)<=0.000000001000 or np.abs(T_03_[2,1]+1)<=0.000000000001: #np.arccos(T_03_[2,1])<=10**(-5):
       
                q2=0
                q1=self.Q0
                q3=self.Q0

                if np.abs(np.abs(T_07[:3,-2].reshape(1,3)@np.array([0,0,1]).reshape(3,1))-1)<=0.00001 and np.linalg.norm(T_07[:2, -1])<=0.00001:
                    q_sol_4567[num_sol][-1]=self.Q0

                count_ += 1
                q_13_this = [q1, q2, q3]
                q_13_this.extend(q_sol_4567[num_sol])
                q_sol.append(q_13_this)

            else:

                q2=np.arccos(T_03_[2,1])
            
                q1=math.atan2(T_03_[1,1]/np.sin(q2),T_03_[0,1]/np.sin(q2))
    
                q3=math.atan2(-T_03_[2,2]/np.sin(q2),-T_03_[2,0]/np.sin(q2))

                if np.abs(np.abs(T_07[:3,-2].reshape(1,3)@np.array([0,0,1]).reshape(3,1))-1)<=0.00001 and np.linalg.norm(T_07[:2, -1])<=0.00001:
                    q1 = self.Q0
                    q_sol_4567[num_sol][-1] = self.Q0

                elif np.abs(np.abs(T_07[:3,-2].reshape(1,3)@T_03_[:3,-2].reshape(3,1))-1)<=0.00001 and np.abs(np.abs(T_07[:3,-2].reshape(1,3)@(T_03_[:3,-1]-T_07[:3,-1]).reshape(3,1))/np.linalg.norm(T_03_[:3,-1]-T_07[:3,-1])-1)<=0.00001:
                    q3=self.Q0
                    q_sol_4567[num_sol][-1] = self.Q0

                count_+=1
                q_13_this=[q1,q2,q3]
                q_13_this.extend(q_sol_4567[num_sol])
                q_sol.append(q_13_this)

                q2=-np.arccos(T_03_[2,1])
            
                q1=math.atan2(T_03_[1,1]/np.sin(q2),T_03_[0,1]/np.sin(q2))
    
                q3=math.atan2(-T_03_[2,2]/np.sin(q2),-T_03_[2,0]/np.sin(q2))

                if np.abs(np.abs(T_07[:3,-2].reshape(1,3)@np.array([0,0,1]).reshape(3,1))-1)<=0.00001 and np.linalg.norm(T_07[:2, -1])<=0.00001:
                    q1 = self.Q0
                    q_sol_4567[num_sol][-1] = self.Q0

                elif np.abs(np.abs(T_07[:3,-2].reshape(1,3)@T_03_[:3,-2].reshape(3,1))-1)<=0.00001 and np.abs(np.abs(T_07[:3,-2].reshape(1,3)@(T_03_[:3,-1]-T_07[:3,-1]).reshape(3,1))/np.linalg.norm(T_03_[:3,-1]-T_07[:3,-1])-1)<=0.00001:
                    q3=self.Q0
                    q_sol_4567[num_sol][-1] = self.Q0
            
                count_+=1
                q_13_this=[q1,q2,q3]
                q_13_this.extend(q_sol_4567[num_sol])
                q_sol.append(q_13_this)



        co=0
        q_sol_final=[]
        for ii in range(len(q_sol)):
            test_count=0
            for jj in range(ii+1, len(q_sol)):
                if np.linalg.norm(np.array(q_sol[ii])-np.array(q_sol[jj])) <= 0.00001:
                    test_count=1
            if test_count==0:
                co+=1
                q_sol_final.append(q_sol[ii])

        limits = [
            {'lower': -2.8973, 'upper': 2.8973},
            {'lower': -1.7628, 'upper': 1.7628},
            {'lower': -2.8973, 'upper': 2.8973},
            {'lower': -3.0718, 'upper': -0.0698},
            {'lower': -2.8973, 'upper': 2.8973},
            {'lower': -0.0175, 'upper': 3.7525},
            {'lower': -2.8973, 'upper': 2.8973}
        ]

        q_sol_final_limited=[]
        # error=[]
        for iii in range(len(q_sol_final)):
            check=0
            for jjj in range(7):
                if q_sol_final[iii][jjj]>=limits[jjj].get('lower') and q_sol_final[iii][jjj]<=limits[jjj].get('upper'):
                    check+=1
            if check==7:
                q_sol_final_limited.append(q_sol_final[iii])
            # p, T = FK().forward(np.array(q_sol_final[iii]))
            # error.append(np.linalg.norm(T - T_07))

        #print(q_sol_final)
        limit_joints=True
        if limit_joints ==True:

            q_sol_ = np.array(q_sol_final_limited)

        else:
            q_sol_ = np.array(q_sol_final)

        # print(q_sol_)
        # Student's code goes in between:

        if len(q_sol_)>0:

            ## DO NOT EDIT THIS PART
            # This will convert your joints output to the autograder format
            q = self.sort_joints(q_sol_)
            # print(q)
            ## DO NOT EDIT THIS PART

        else:
            print("Empty Output")
            q = np.array([]).reshape(0, 7)

        # if np.linalg.norm(T_07[:3,-1].reshape(3)-np.array([0,0,self.d1]))>np.sqrt([self.a3**2+self.d3**2])+np.sqrt([self.a3**2+self.d5**2])+np.sqrt([self.a6**2+self.d7**2]):
        #     q = np.array([]).reshape(0, 7)

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
        """
        wrist_pos = []

        T_07=self.get_transformation(target)



        T_70=np.linalg.inv(T_07)

        t_72=T_70[:3,:3]@np.array([0,0,self.d1]).reshape(3,1)+T_70[:3,-1].reshape(3,1)

        wrist_pos=t_72.reshape(3)

        return wrist_pos 

    def ik_pos(self, wrist_pos):
        """
        Solves IK position problem on the joint 4, 6, 7 
        Args: 
            wrist_pos: 3x1 numpy array of the position of the wrist center in frame 7
        Returns:
             joints_467 = nx3 numpy array of all joint angles of joint 4, 6, 7
        """
        joints_467 = []

        t_72=wrist_pos

        q7=math.atan2(-t_72[1],t_72[0])+pi/4

        q7=self.dis_ambiguity(q7)

        t_72_=np.array([[np.cos(q7-pi/4),-np.sin(q7-pi/4),0],
                   [np.sin(q7-pi/4),np.cos(q7-pi/4),0],
                   [0,0,1]])@t_72.reshape(3,1)
        #print(t_72_)

        angle1=math.atan2(self.a3,self.d5)
        l1=np.sqrt(self.a3**2+self.d5**2)
        angle2=math.atan2(self.d5,self.a3)+math.atan2(self.d3,self.a3)
        l2=np.sqrt(self.a3**2+self.d3**2)

        # possibility 1
        t_72_updated=np.array([t_72_[0],t_72_[-1]])+np.array([self.a6,self.d7]).reshape(2,1)

        q_sol_4567=[]
        count=0

        l3=np.linalg.norm(t_72_updated)

        if l1+l2>l3 and l1-l2<l3:

            q4=angle2-np.arccos((-l3**2+l1**2+l2**2)/(2*l1*l2))

            q4=self.dis_ambiguity(q4)

            theta=math.atan2(-t_72_updated[0],t_72_updated[1])-np.arccos((-l2**2+l1**2+l3**2)/(2*l1*l3))
            q6=angle1+theta

            q6=self.dis_ambiguity(q6)

            count+=1
            q_sol_4567.append([q4,0,q6,q7])

            q4=angle2+np.arccos((-l3**2+l1**2+l2**2)/(2*l1*l2))
            q4=self.dis_ambiguity(q4)

            theta=math.atan2(-t_72_updated[0],t_72_updated[1])+np.arccos((-l2**2+l1**2+l3**2)/(2*l1*l3))
            q6=angle1+theta
            q6=self.dis_ambiguity(q6)

            count+=1
            q_sol_4567.append([q4,0,q6,q7])

        elif np.abs(l1+l2-l3)<10**(-5):
            count += 1
            q_sol_4567.append([math.atan2(self.d3,self.a3)+math.atan2(self.d5,self.a3)-pi, 0, pi-math.atan2(self.d5,self.a3) + math.atan2(self.d7,self.a6) , pi/4])


        # possibility 2
    
        q7=IK().dis_ambiguity(q7-pi)

        t_72_=np.array([[np.cos(q7-pi/4),-np.sin(q7-pi/4),0],
                   [np.sin(q7-pi/4),np.cos(q7-pi/4),0],
                   [0,0,1]])@t_72.reshape(3,1)

        t_72_updated=np.array([t_72_[0],t_72_[-1]])+np.array([self.a6,self.d7]).reshape(2,1)

        l3=np.linalg.norm(t_72_updated)

        if np.abs(t_72_[0])>10**(-6) and l1+l2>l3 and l1-l2<l3 :

            q4=angle2-np.arccos((-l3**2+l1**2+l2**2)/(2*l1*l2))

            q4=self.dis_ambiguity(q4)
    
            theta=math.atan2(-t_72_updated[0],t_72_updated[1])-np.arccos((-l2**2+l1**2+l3**2)/(2*l1*l3))
            q6=angle1+theta

            q6=self.dis_ambiguity(q6)

            count+=1
            q_sol_4567.append([q4,0,q6,q7])

            q4=angle2+np.arccos((-l3**2+l1**2+l2**2)/(2*l1*l2))
            q4=self.dis_ambiguity(q4)

            theta=math.atan2(-t_72_updated[0],t_72_updated[1])+np.arccos((-l2**2+l1**2+l3**2)/(2*l1*l3))
            q6=angle1+theta
            q6=self.dis_ambiguity(q6)

            count+=1
            q_sol_4567.append([q4,0,q6,q7])

        elif np.abs(l1+l2-l3)<10**(-5):
            count += 1
            q_sol_4567.append([math.atan2(self.d3,self.a3)+math.atan2(self.d5,self.a3)-pi, 0, pi-math.atan2(self.d5,self.a3) + math.atan2(self.d7,self.a6) , pi/4])


        joints_467=np.array(q_sol_4567)
        
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
        joints_123 = []
        
        return joint_123
    
    def sort_joints(self, q, col=0):
        """
        Sort the joint angle matrix by ascending order 
        Args: 
            q: nx7 joint angle matrix 
        Returns: 
            q_as = nx7 joint angle matrix in ascending order 
        """
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
        return q_as


    def get_transformation(self, target):

        R_07=target['R']

        t_07=target['t']

        T_07=np.eye(4)

        T_07[:3,:3]=R_07

        T_07[:3,-1]=t_07.reshape(3)

        return T_07

    def dis_ambiguity(self,q):

        if q>pi:
            q-=2*pi
        elif q<=-pi:
            q+=2*pi
        return q
        


if __name__ == '__main__':

    # T_07 = np.array(
    #     [[-0.7712 ,   0.5455 ,  -0.3283,   -0.0234], [0.0119,   -0.5032,   -0.8641,   -0.7754], [-0.6365,   -0.6703,    0.3816,    0.6880],
    #      [0, 0, 0, 1]])
    # target = {"R": T_07[:3, :3], "t": T_07[:3, -1].reshape(3,1)}
    # q_sol=IK().panda_ik(target)
    # print(q_sol)

    limits = [
        {'lower': -2.8973, 'upper': 2.8973},
        {'lower': -1.7628, 'upper': 1.7628},
        {'lower': -2.8973, 'upper': 2.8973},
        {'lower': -3.0718, 'upper': -0.0698},
        {'lower': -2.8973, 'upper': 2.8973},
        {'lower': -0.0175, 'upper': 3.7525},
        {'lower': -2.8973, 'upper': 2.8973}
    ]

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

    while True:
        q_in=random.rand(7)*2*pi-pi
        q_in[4]=0
        # q_in[1]=-math.atan2(0.0825,0.316)
        # q_in[3]=-math.atan2(0.0825, 0.316)
        ok=0
        for ii in range(7):
            if q_in[ii]>=limits[ii].get('lower') and q_in[ii]<=limits[ii].get('upper'):
                ok+=1
        if ok==7:
            break


    q_in=np.array([-2.5, -1.7, -1.7, -2.1, 0, 0.3, -2.8])

    q_in=np.array([1.3, -1.2,  0.3,-2.84079278 , 0, 2.6 ,-0.7])

    q_in=np.array([0, pi/2-math.atan2(a3,d3), 0, math.atan2(d3,a3) +math.atan2(d5,a3)-pi, 0, pi-math.atan2(d5,a3) + math.atan2(d7,a6) , pi/4])

    q_in=np.array([-pi/3, -0.60546154,0,-2.31761045,0,0.8121489,pi/4])

    q_in = np.array([0, -1.50546154, 0, -2.31761045, 0, 0.8121489, 0])

    q_in=np.array([2 , 0, pi/2 , -0.07, 0, pi, 0])

    q_in=np.array([0, -math.atan2(a3,d3), 0, math.atan2(d3,a3) +math.atan2(d5,a3)-pi, 0, pi-math.atan2(d5,a3) + math.atan2(d7,a6) , pi/4])


    p, T_gt = FK().forward(q_in)

    # T_gt=np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0.55],[0,0,0,1]])

    target = {"R": T_gt[:3, :3], "t": T_gt[:3, -1].reshape(3, 1)}

    q_sol = IK().panda_ik(target)

    num_sol = len(q_sol)
    print(num_sol)

    Trans_error = np.zeros(shape=(num_sol))
    q_error = np.zeros(shape=(num_sol))
    for id in range(num_sol):
        p, T_solved = FK().forward(q_sol[id])
        Trans_error[id]=np.linalg.norm(np.array(T_gt-T_solved))
        q_error[id]=np.linalg.norm(q_sol[id]-q_in)

    print(q_sol)
    print("Max Transformation Error:", max(Trans_error))
    print("q-error:", min(q_error))


    # q_err=[]
    # for i in range(len(q_sol)):
    #     q_err.append(np.linalg.norm(q_sol[i]-q_in))
    #
    # min(q_err)

        


    
