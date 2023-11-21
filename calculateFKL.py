import numpy as np
from math import pi

"""
Author: Lei Sun (ID: 61330435)
"""

class FK():
    """
    This is the standard FK function (class) that solves the end effector transformation given robot configuration
    Author: Lei Sun
    Note that this class only contains FK-related subroutines of the manipulation algorithm
    Please see "final.py" for the full object manipulation algorithm
    """

    def __init__(self):

        # TODO: you may want to define geometric parameters here that will be
        # useful in computing the forward kinematics. The data you will need
        # is provided in the lab handout

        pass

    def forward(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        jointPositions -8 x 3 matrix, where each row corresponds to a rotational joint of the robot or end effector
                  Each row contains the [x,y,z] coordinates in the world frame of the respective joint's center in meters.
                  The base of the robot is located at [0,0,0].
        T0e       - a 4 x 4 homogeneous transformation matrix,
                  representing the end effector frame expressed in the
                  world frame
        """

        # Your Lab 1 code starts here:
        
        ### The following codes are written by Lei Sun independently:
        
        q_base=np.array([0,0,0,-np.pi/2,0,np.pi/2,np.pi/4])
        
        q_this=q-q_base
        
        # We tend to use method 1 by default
        # If you want to use the DH convention method, please change DH_use into True
        DH_use=False
        
        
        # First set sin and cos values of the 7 joints as two vectors for the convenience of operation later
        S=np.array([np.sin(q_this[0]),np.sin(q_this[1]),np.sin(q_this[2]),np.sin(q_this[3]),np.sin(q_this[4]),np.sin(q_this[5]),np.sin(q_this[6])])
        C=np.array([np.cos(q_this[0]),np.cos(q_this[1]),np.cos(q_this[2]),np.cos(q_this[3]),np.cos(q_this[4]),np.cos(q_this[5]),np.cos(q_this[6])])
        
        ## Method 1. Traditional Homogeneous Transformation:
        T_01=np.array([[C[0],-S[0],0,0],[S[0],C[0],0,0],[0,0,1,0.141],[0,0,0,1]])
        T_12=np.array([[C[1],-S[1],0,0],[0,0,1,0],[-S[1],-C[1],0,0.192],[0,0,0,1]])
        T_23=np.array([[C[2],-S[2],0,0],[0,0,-1,-0.195],[S[2],C[2],0,0],[0,0,0,1]])
        T_34=np.array([[C[3],-S[3],0,0.0825],[0,0,-1,0],[S[3],C[3],0,0.121],[0,0,0,1]])
        T_45=np.array([[0,0,1,0.125],[-C[4],S[4],0,0.0825],[-S[4],-C[4],0,0],[0,0,0,1]])
        T_56=np.array([[C[5],-S[5],0,0],[0,0,-1,0.015],[S[5],C[5],0,0.259],[0,0,0,1]])
        T_67=np.array([[0,0,1,0.051],[C[6],-S[6],0,0.088],[S[6],C[6],0,0.015],[0,0,0,1]])
        T_7e=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0.159],[0,0,0,1]])
        
        
        ## Method 2. DH Convention:
        if DH_use==True:
            T_01=np.array([[C[0],-S[0],0,0],[S[0],C[0],0,0],[0,0,1,0.141],[0,0,0,1]])
            T_12_1=np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,0.192],[0,0,0,1]])
            T_12_2=np.array([[C[1],-S[1],0,0],[S[1],C[1],0,0],[0,0,1,0],[0,0,0,1]])
            T_12=T_12_1@T_12_2
            T_23_1=np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]])
            T_23_2=np.array([[C[2],-S[2],0,0],[S[2],C[2],0,0],[0,0,1,0.195],[0,0,0,1]])
            T_23=T_23_1@T_23_2
            T_34_1=np.array([[1,0,0,0.0825],[0,0,-1,0],[0,1,0,0.121],[0,0,0,1]])
            T_34_2=np.array([[C[3],-S[3],0,0],[S[3],C[3],0,0],[0,0,1,0],[0,0,0,1]])
            T_34=T_34_1@T_34_2
            T_45_1=np.array([[0,0,1,0],[-1,0,0,0.0825],[0,-1,0,0],[0,0,0,1]])
            T_45_2=np.array([[C[4],-S[4],0,0],[S[4],C[4],0,0],[0,0,1,0.125],[0,0,0,1]])
            T_45=T_45_1@T_45_2
            T_56_1=np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0.259],[0,0,0,1]])
            T_56_2=np.array([[C[5],-S[5],0,0],[S[5],C[5],0,0],[0,0,1,-0.015],[0,0,0,1]])
            T_56=T_56_1@T_56_2
            T_67_1=np.array([[0,0,1,0],[1,0,0,0],[0,1,0,0.015],[0,0,0,1]])
            T_67_2=np.array([[C[6],-S[6],0,0.088],[S[6],C[6],0,0],[0,0,1,0.051],[0,0,0,1]])
            T_67=T_67_1@T_67_2
            T_7e=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0.159],[0,0,0,1]])
            

        # Set empty 8x3 matrix to store the joint coordinates
        jointPositions = np.zeros((8,3))
        
        # Then compute the transformation matrix and joint coordinates in sequence (from frame 0 to the end effector)
        jointPositions[0,:]=T_01[0:3,-1].T
        T_02=T_01@T_12
        jointPositions[1,:]=T_02[0:3,-1].T
        T_03=T_02@T_23
        jointPositions[2,:]=T_03[0:3,-1].T
        T_04=T_03@T_34
        jointPositions[3,:]=T_04[0:3,-1].T
        T_05=T_04@T_45
        jointPositions[4,:]=T_05[0:3,-1].T
        T_06=T_05@T_56
        jointPositions[5,:]=T_06[0:3,-1].T
        T_07=T_06@T_67
        jointPositions[6,:]=T_07[0:3,-1].T
        T0e=T_07@T_7e
        jointPositions[7,:]=T0e[0:3,-1].T
        

        # Your code ends here

        return jointPositions, T0e

    # feel free to define additional helper methods to modularize your solution for lab 1

    
    # This code is for Lab 2, you can ignore it ofr Lab 1
    def get_axis_of_rotation(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        axis_of_rotation_list: - 3x7 np array of unit vectors describing the axis of rotation for each joint in the
                                 world frame

        """
        # STUDENT CODE HERE: This is a function needed by lab 2

        return()
    
    def compute_Ai(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        Ai: - 4x4 list of np array of homogenous transformations describing the FK of the robot. Transformations are not
              necessarily located at the joint locations
        """
        # STUDENT CODE HERE: This is a function needed by lab 2

        return()
    
if __name__ == "__main__":

    fk = FK()

    # matches figure in the handout
    q = np.array([0,0,0,-pi/2,0,pi/2,pi/4])

    joint_positions, T0e = fk.forward(q)
    
    print("Joint Positions:\n",joint_positions)
    print("End Effector Pose:\n",T0e)
