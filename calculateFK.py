# calculateFK.py
import numpy as np
from math import pi
from core.interfaces import ObjectDetector
from core.interfaces import ArmController

class FK():
    """
    This is the FK function (class) that employs the vision system to estimate object pose
    Author: Jiaqi Lian
    Note that this class only contains FK-related subroutines of the manipulation algorithm
    Please see "final.py" for the full object manipulation algorithm
    """

    def __init__(self):

        # TODO: you may want to define geometric parameters here that will be
        # useful in computing the forward kinematics. The data you will need
        # is provided in the lab handout
        self.detector = ObjectDetector()
        self.arm = ArmController()

        pass

    def DH_to_trans(self,DH):
        """
        DH parameters input: an array include [a,alpha,d,theta]
        output: transformation matrix
        """
        a,alpha,d,theta = DH
        single_trans = np.array([[np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
                        [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
                        [0, np.sin(alpha), np.cos(alpha), d],
                        [0, 0, 0, 1]])
        return single_trans

    def forward(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q1, q2, q3, q4, q5, q6, q7]

        OUTPUTS:
        jointPositions -8 x 3 matrix, where each row corresponds to a rotational joint of the robot or end effector
                  Each row contains the [x,y,z] coordinates in the world frame of the respective joint's center in meters.
                  The base of the robot is located at [0,0,0].
        T0e       - a 4 x 4 homogeneous transformation matrix,
                  representing the end effector frame expressed in the
                  world frame
        """

        # Your Lab 1 code starts here

        #jointPositions = np.zeros((8,3))
        jointPositions = []
        q = np.array(q)
        q1, q2, q3, q4, q5, q6, q7 = q
        DH_paras = np.array([[0,        np.pi/2,    0.333,      np.pi+q1                ],
                             [0,        np.pi/2,    0,          np.pi+q2                ],
                             [0.0825,   np.pi/2,    0.316,      q3                      ],
                             [0.0825,   np.pi/2,    0,          np.pi/2+q4-(-np.pi/2)   ],
                             [0,        0,          0.125,      0                       ],
                             [0,        np.pi/2,    0.259,      q5                      ],
                             [0,        np.pi,      0.015,      0                       ],
                             [0.088,    np.pi/2,    0.015,     -np.pi/2+q6-np.pi/2      ],
                             [0,        0,          0.051,      0                       ],
                             [0,        0,          0.159,      q7-np.pi/4              ]])

        trans = np.array([self.DH_to_trans(DH_paras[0]),
                          self.DH_to_trans(DH_paras[1]),
                          self.DH_to_trans(DH_paras[2]),
                          self.DH_to_trans(DH_paras[3]),
                          self.DH_to_trans(DH_paras[4]),
                          self.DH_to_trans(DH_paras[5]),
                          self.DH_to_trans(DH_paras[6]),
                          self.DH_to_trans(DH_paras[7]),
                          self.DH_to_trans(DH_paras[8]),
                          self.DH_to_trans(DH_paras[9])])

        """
        vn_m: joint n in frame m
        """
        #Joint 1
        jointPositions.append(np.array([0,0,0.141]))

        #Joint 2
        v2_1 = np.array([0,0,0,1])
        v2_0 = trans[0]@v2_1.T
        jointPositions.append(v2_0[0:3])

        #Joint 3
        v3_2 = np.array([0,0,0.195,1])
        v3_0 = trans[0]@trans[1]@v3_2.T
        jointPositions.append(v3_0[0:3])

        #joint 4
        v4_3 = np.array([0,0,0,1])
        v4_0 = trans[0]@trans[1]@trans[2]@v4_3.T
        jointPositions.append(v4_0[0:3])

        #joint 5
        v5_5 = np.array([0,0,0,1])
        v5_0 = trans[0]@trans[1]@trans[2]@trans[3]@trans[4]@v5_5.T
        jointPositions.append(v5_0[0:3])

        #joint6
        v6_7 = np.array([0,0,0,1])
        v6_0 = trans[0]@trans[1]@trans[2]@trans[3]@trans[4]@trans[5]@trans[6]@v6_7.T
        jointPositions.append(v6_0[0:3])

        #joint 7
        v7_9 = np.array([0,0,0,1])
        v7_0 = trans[0]@trans[1]@trans[2]@trans[3]@trans[4]@trans[5]@trans[6]@trans[7]@trans[8]@v7_9.T
        jointPositions.append(v7_0[0:3])

        #end effector
        ve_e = np.array([0,0,0,1])
        T0e = trans[0]@trans[1]@trans[2]@trans[3]@trans[4]@trans[5]@trans[6]@trans[7]@trans[8]@trans[9]
        ve_0 = T0e@ve_e.T
        jointPositions.append(ve_0[0:3])

        jointPositions = np.array(jointPositions)
        # Your code ends here

        return jointPositions, T0e

    def get_camera_T0c(self,q):
        """
        This functions estimates the camera pose in base frame
        :param q: input configuration
        :return: base-to-camera pose
        """
        _,T0e = self.forward(q)
        Tec =  self.detector.get_H_ee_camera()
        T0c = T0e@Tec
        cam_pos = T0c[:,3].flatten()[0:3]
        return cam_pos,T0c

    def scan_blocks(self):
        """
        This function scans the blocks with the vision system and returns the data regarding these blocks
        :return: block names, poses and positions
        """
        q = self.arm.get_positions() # scan current angle positions of panda

        cam_pos,T0c = self.get_camera_T0c(q)
        info_blocks = self.detector.get_detections()
        num_blocks = len(info_blocks)
        name = []
        Tcb = []
        T0b = []
        pos_blocks = []


        if num_blocks != 0:
            for name_i, Tcb_i in info_blocks:
                T0b_i = T0c@Tcb_i
                pos_i = T0b_i[:,3].flatten()[0:3]

                name.append(name_i)
                T0b.append(T0b_i)
                pos_blocks.append(pos_i)

        name = np.asarray(name)
        pos_blocks = np.asarray(pos_blocks)
        T0b = np.asarray(T0b)
        return name, T0b, pos_blocks
