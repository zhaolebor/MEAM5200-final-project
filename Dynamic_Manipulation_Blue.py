import sys
import numpy as np
from copy import deepcopy
from math import pi
import time

import rospy
# Common interfaces for interacting with both the simulation and real environments!
from core.interfaces import ArmController
from core.interfaces import ObjectDetector

# for timing that is consistent with simulation or real time as appropriate
from core.utils import time_in_seconds
from calculateFKL import FK
from calculateIK6L import IK


def Dynamic_Manipulation_Blue(hardware=True, arm_speed=0.2, max_itr_of_grabbing=3, robot_to_table_dis=0.99, min_gripper_pos=0.02, grip_time=1, height_offset=0, time_of_sleep=12, time_of_wait=10, picking_height_offset=0, dynamics_x_offset=0, dynamics_y_offset=0, dynamics_z_offset=0):
    """
    This function is the "Ambusher" dynamic manipulation method (for blue side).
    Author: Lei Sun
    Note that this function is not used in final competition. Please refer to "final.py" for the adopted blue-side code.
    :param hardware: boolean, whether on hardware or not
    :param arm_speed: speed of arm, 0.2 by default (not used anymore, please ignore it)
    :param max_itr_of_grabbing: maximum iterations of gripping
    :param robot_to_table_dis: 0.99 meters by default (do not change this on hardware)
    :param min_gripper_pos: the threshold of gripper position that differentiate having gripped a block or nat not having gripped it
    :param grip_time: time of gripping in each iteration
    :param height_offset: if you have run and stack the static 4 block, set it to 0.2; if not, set it to 0.2
    :param time_of_sleep: time for evaluating good-enough target block
    :param time_of_wait: time for waiting for the target block
    :param picking_height_offset: z-axis offset of picking position, positive-up, negative-down (0 by default)
    :param dynamics_x_offset: stacking offsets in x-axis direction
    :param dynamics_y_offset: stacking offsets in y-axis direction
    :param dynamics_z_offset: stacking offsets in z-axis direction
    :return: the number of blocks successfully picked
    """

    # Set up all basic info and parameters
    arm = ArmController()
    detector = ObjectDetector()
    fk=FK()
    H_ee_camera = detector.get_H_ee_camera()

    # Set the stacking configuration and compute its pose using FK
    q1 = np.array([0.2,0.1,0,-pi/2.5,0,0.1+pi/2.5,pi/4])
    q2=q1+np.array([-pi/7,0.1,0,0,0,0.1,0])
    p2, T2=fk.forward(np.array(q2))

    # Update the stacking pose (since there is 4 static blocks stacked already)
    T2[2,-1]+=height_offset-0.34

    # Set an initial "naive" picking pose
    T_process=np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
    
    T_obj_to_end=np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])

    # Count the number of picked-and-placed blocks
    picked_count=0

    # Repeat dynamic manipulation by max_itr_of_grabbing times
    for itr_of_grabbing in range(max_itr_of_grabbing):
        print("Dynamic itr: "+str(itr_of_grabbing+1))

        # Set default scanning configuration, compute its pose using FK, and move to it
        q3 = np.array([-pi/2,0.25,0,-pi/3,0,0.25+pi/3,-pi/4])
        p, T=fk.forward(q3)
        arm.safe_move_to_position(q3)

        # Open the gripper
        arm.exec_gripper_cmd(0.1, 1)

        # Get end-to-cam pose
        H_ee_camera = detector.get_H_ee_camera()

        # Repeatedly scanning the blocks on the turntable
        while True :
            # First set valid_obj_get as False
            valid_obj_get=False # valid_obj_get to check if valid block is detected
            # Go through all the detected blocks
            for (name, pose) in detector.get_detections():
                Pose = np.array(pose)
                # Rectify and adjust the block pose
                if Pose[1,-1]<-0.07 and np.max(np.abs(Pose[2,:3]))>0.99 and Pose[2,-1]<0.6:
                    if (np.abs(Pose[2,1])>0.99):
                        Pose[:3,:3]=np.concatenate((Pose[:3,2].reshape(3,1),Pose[:3,0].reshape(3,1),Pose[:3,1].reshape(3,1)),axis=1)
                    if (np.abs(Pose[2,0])>0.99):
                        Pose[:3,:3]=np.concatenate((Pose[:3,1].reshape(3,1),Pose[:3,2].reshape(3,1),Pose[:3,0].reshape(3,1)),axis=1)
                    pose_e = H_ee_camera @ Pose
                    if (pose_e[2,2]<0):
                        pose_e = pose_e @ T_process
                    T_O=T @ pose_e

                    # Compute the block radius w.r.t. the turntable center
                    radius=np.linalg.norm(T_O[:2,-1].reshape(2)-np.array([0,-robot_to_table_dis]))

                    # Compute the central degree of this block w.r.t. the turntable center
                    deg=np.arctan(np.abs(T_O[0,-1]/(T_O[1,-1]+robot_to_table_dis)))*180/pi;
                    
                    # Predict the arrival position of the block
                    T_O[1,-1]=-robot_to_table_dis+radius
                    T_O[0,-1]=0
                    # if hardware==False:
                    #     T_O[2,-1]+=0.05  ## PLEASE comment this line if you run the code on hardware!!!

                    # Set the waiting and catching pose (known as the "lie-down" pose) by rotating the end effector
                    # along the y-axis (in base frame) by 72 degrees
                    T_O[:3,:3]= np.array([[1,0,0],[0,-1,0],[0,0,-1]]) @ np.array([[np.cos(pi/2.5),0,np.sin(pi/2.5)],[0,1,0],[-np.sin(pi/2.5),0,np.cos(pi/2.5)]])
                    
                    # Solve IK for the catching pose
                    target={'R':T_O[:3,:3], 't': T_O[:3,-1].reshape(3)}
                    q_sol=IK().panda_ik(target)
                    sol_idx=-1

                    # Check if IK has solution
                    if len(q_sol)>1:
                        qq=arm.get_positions()
                        # Check if the solutions are eligible
                        for i_sol in range(len(q_sol)):
                            qq[0],qq[2],qq[3],qq[6]=q_sol[i_sol,0],q_sol[i_sol,2],q_sol[i_sol,3],q_sol[i_sol,6]
                            if  q_sol[i_sol,0]>0 and q_sol[i_sol,0]<pi/2 :
                                sol_idx=i_sol
                    # If valid IK solutions can be solved and the time to arrival is qualified, jump out the while loop
                    if sol_idx>-1 and deg>=3.6*(time_of_sleep+2) and deg<3.6*(time_of_sleep+4.2)  and T_O[1,-1]>-0.8:
                        print("Radius of the block: "+str(radius))
                        print("Degree of the block: "+str(deg))
                        valid_obj_get=True
                        break
            if valid_obj_get:
                break
        print("A valid block found!")
        print(name,'\n',pose)
        print("Catching pose: ")
        print(T_O)

        # If a valid block is detected
        if valid_obj_get:

            # First get the actual picking pose
            q_sol_act=q_sol[sol_idx,:].reshape(7)
            print(q_sol_act)
            T_=np.array(T_O)

            # Set and move to the first pre-picking configuration
            qqq=np.array([0.2 , -0.25 , -1.88091563e+00 ,-1.06849577e+00 ,7.70042028e-06 , 1.29720153e+00 , -9.21888541e-01])
            arm.safe_move_to_position(qqq)

            # Set and move to the second pre-picking configuration
            dif=np.abs(-0.5-T_[1,-1])
            T_[1,-1]+=dif
            T_[2,-1]+=picking_height_offset
            # Solve IK
            target={'R':T_[:3,:3], 't': T_[:3,-1].reshape(3)}
            q_sol=IK().panda_ik(target)
            print(q_sol)
            if len(q_sol)>1:
                for j_sol in range(len(q_sol)):
                    if q_sol[j_sol,0]<pi/2 and q_sol[j_sol,0]>-pi/4 :
                        q_sol_=q_sol[j_sol,:].reshape(7)
                        break
            else:
                q_sol_=q_sol[0,:].reshape(7)
            print("q_sol_")
            #print(q_sol_)
            q_sol_=np.array([-0.07191352, -1.24217724 ,-1.7681778 , -2.23668147 , 0.    ,      2.16693695, -1.00550544])
            arm.safe_move_to_position(q_sol_)
            q_sol_=arm.get_positions()

            # Old version, please ignore this code block!
            T_[1,-1]-=dif
            target={'R':T_[:3,:3], 't': T_[:3,-1].reshape(3)}
            q_sol=IK().panda_ik(target)
            q_now=q_sol_
            norm_min=100
            for i in range(len(q_sol)):
                if np.linalg.norm(q_sol[i,:3]-q_now[:3])<norm_min:
                    norm_min=np.linalg.norm(q_sol[i,:3]-q_now[:3])
                    q_sol_=q_sol[i,:].reshape(7)

            # Move to the "lie-down" catching configuration
            arm.safe_move_to_position(q_sol_act)
            arm.safe_move_to_position(q_sol_act)
            arm.safe_move_to_position(q_sol_act)
            q_sol_=arm.get_positions()
            
            catched=False
            # Only grip once
            grip_time = 1
            # Wait for a preset period of time and close the gripper to grip the block
            for trials in range(grip_time):
                time.sleep(time_of_wait)
                arm.exec_gripper_cmd(0.02, 75)
                    
            # Old version, please ignore this code block!
            T_[2,-1]+=0.25
            T_[1,-1]+=0.1
            target={'R':T_[:3,:3], 't': T_[:3,-1].reshape(3)}
            q_sol=IK().panda_ik(target)
            q_now=q_sol_
            norm_min=1000
            for i in range(len(q_sol)):
                if np.linalg.norm(q_sol[i,:]-q_now[:])<norm_min:
                    norm_min=np.linalg.norm(q_sol[i,:]-q_now)
                    q_sol_=q_sol[i,:].reshape(7)
            print(q_sol_)

            # Elevate the end effector (move to the first pre-stacking pose)
            Qq=np.array([0.27210466, -1.21641407, -1.46141844, -1.60646526,  0,          1.86652871, -0.58623534])
            arm.safe_move_to_position(Qq)
            q_sol_=arm.get_positions()
            q_now=np.array(q_sol_)

            # Set stacking offsets in x,y,z-axis directions empirically
            T2[0,-1]+=dynamics_x_offset
            T2[1,-1]+=dynamics_y_offset
            T2[2,-1]+=dynamics_z_offset

            # Check the status of gripper to determine if the block is caught successfully
            res=arm.get_gripper_state()
            grip_pos=res.get('position')
            print(grip_pos)
            if np.abs(grip_pos[0])+np.abs(grip_pos[1])>min_gripper_pos: #min(np.abs(res.get('position')))>min_gripper_pos:
                catched=True
            
            # If the block is caught successfully, then move to the stacking position
            if catched :
                # Update the stacking pose height and compute the stacking configuration using IK
                T2[2,-1]+=0.05
                T2[:3,:3] = np.array([[1,0,0],[0,np.cos(pi/2-pi/2.5),-np.sin(pi/2-pi/2.5)],[0,np.sin(pi/2-pi/2.5), np.cos(pi/2-pi/2.5)]]) @ np.array([[0,-1,0],[0,0,-1],[1,0,0]])
                target2={'R':T2[:3,:3], 't': T2[:3,-1].reshape(3)}
                q_place=IK().panda_ik(target2)
                if itr_of_grabbing==-1:
                    target2_int={'R':T2[:3,:3], 't': T2[:3,-1].reshape(3)+np.array([0,0,0.35])}
                    q_place_int=IK().panda_ik(target2_int)
                    norm_min=100
                    q_now=arm.get_positions()
                    for i in range(len(q_place_int)):
                        if np.linalg.norm(q_place_int[i,:]-q_now[:])<norm_min:
                            norm_min=np.linalg.norm(q_place_int[i,:]-q_now)
                            q_place_int_=q_place_int[i,:].reshape(7)
                    print(q_place_int_)
                q_place_int_=np.array([ 0.72369829 , 0.85046871, -1.82233887, -0.80144596 , 0.  ,        1.6283565, 1.65705788])

                # Compute the least-energy stacking configuration
                norm_min=100
                q_now=arm.get_positions()
                for i in range(len(q_place)):
                    if np.linalg.norm(q_place[i,:]-q_now[:])<norm_min:
                        norm_min=np.linalg.norm(q_place[i,:]-q_now)
                        q_place_=q_place[i,:].reshape(7)
                print(q_place_)

                # Move to the second pre-stacking pose
                arm.safe_move_to_position(q_place_int_)
                arm.safe_move_to_position(q_place_)

                # Open the gripper to release the block
                arm.exec_gripper_cmd(0.099, 1)

                # Move to the intermediate pose to prepare for the next block
                arm.safe_move_to_position(q_place_int_)
                arm.safe_move_to_position(q_place_int_+np.array([0,-0.36,0,0,0,0,0]))
                picked_count+=1

                # Check if the desired number of blocks have been picked and stacked (if yes, terminate the algorithm)
                if picked_count>=5:
                    break
                
