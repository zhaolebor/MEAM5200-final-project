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
from Dynamic_Manipulation_Red import Dynamic_Manipulation_Red
from Dynamic_Manipulation_Blue import Dynamic_Manipulation_Blue

# Set algorithm-related parameters for the blue-team code
idle_grip_width = 0.1
block_width = 0.05
force = 75
turntable_offset = 0.99
dynamic_observe_y = 0.75
dynamic_observe_z = 0.4
radian_threshold = 0.50
gripper_threshold = 0.025
enable_hard = False
sleep_hard = 0.2

"""
The following functions are for the blue-side use only
"""
def get_farthest_block(team, name_block, name, pos_blocks):
    """
    This functions gets the farthest block among all the blocks given
    :param team: team color (red or blue)
    :param name_block: input name of the block
    :param name: all the names of blocks
    :param pos_blocks: positions of all input blocks
    :return: the index of the farthest block
    """
    farthest = -1
    # Go through all the blocks
    for i in range(len(name)):
        # If the block has the same name as the input block, store it
        if name[i] == name_block:
            farthest = i
    # If no block has the same name as name_block, then get the block that is the farthest away
    if farthest == -1:
        if team == 'red':
            farthest = np.argmin(pos_blocks[:, 0])
        else:
            farthest = np.argmax(pos_blocks[:, 0])

    return farthest

def dynamic_prediction(team, start_T, omega, time):
    """
    This function predicts the pose of the block when it arrives at the catching pose and also estimates its radius
    w.r.t. the turntable
    :param team: team color (red or blues)
    :param start_T: the start pose of the block
    :return: the arrival pose of the block and its radius w.r.t. the turntable
    """
    x = abs(start_T[0, 3])
    # Get the absolute y-axis value of the block w.r.t. the turntable center
    if team == 'red':
        y = abs(turntable_offset - start_T[1, 3])
    else:
        y = abs(-turntable_offset - start_T[1, 3])
    # Compute the radius of this block w.r.t. the turntable center
    radius = np.sqrt(x ** 2 + y ** 2)
    end_T = start_T.copy()
    end_T[0, 3] = 0
    # Obtain the full block pose when it arrives at the picking position
    if team == 'red':
        end_T[1, 3] = turntable_offset - radius
    else:
        end_T[1, 3] = -turntable_offset + radius
    end_T[2, 3] = 0.224
    if team == 'red':
        end_T[0:3, 0:3] = np.array([[-1,0,0],[0,1,0],[0,0,-1]])
    else:
        end_T[0:3, 0:3] = np.array([[1,0,0],[0,-1,0],[0,0,-1]])

    return end_T, radius


def check_grip(arm, threshold):
    """
    This function checks if the gripper has caught the block.
    :param arm: the arm class
    :param threshold: threshold to differentiate if the block is gripped
    :return: true for block gripped, false for not gripped
    """
    gripper_state = arm.get_gripper_state()
    # Compute the distance between 2 gripper fingers
    actual_gripper_width = sum(gripper_state['position'])
    print(gripper_state)
    # Check the finger distance to determine if the block is gripped successfully
    if actual_gripper_width < threshold:
        return False
    else:
        return True


def singularity(team, q, T_block):
    """
    This function prevents degenerate cases in (near) singularity w.r.t. robot joint 2
    :param team: team color (red or blue)
    :param q: the current config
    :param T_block: the current block pose
    :return: the updated configuration that is not degenerate when encountering singularity
    """
    # These are the operations for the situation where joint 2 is too close to 0 (singularity)
    x = abs(T_block[0,3])
    y = abs(T_block[1,3])
    angle = np.arctan(y/x)
    if team == 'blue':
        q[0] = angle / 2
        q[2] = angle / 2
    else:
        q[0] = -angle / 2
        q[2] = -angle / 2
    return q


"""
The main function for the Pick-and-Place challenge:
"""
if __name__ == "__main__":
    try:
        team = rospy.get_param("team") # 'red' or 'blue'
    except KeyError:
        print('Team must be red or blue - make sure you are running final.launch!')
        exit()

    rospy.init_node("team_script")
    arm = ArmController()
    detector = ObjectDetector()

    start_position = np.array([-0.01779206, -0.76012354,  0.01978261, -2.34205014, 0.02984053, 1.54119353+pi/2, 0.75344866])
    arm.safe_move_to_position(start_position) # on your mark!

    print("\n****************")
    if team == 'blue':
        print("** BLUE TEAM  **")
    else:
        print("**  RED TEAM  **")
    print("****************")
    input("\nWaiting for start... Press ENTER to begin!\n") # get set!
    print("Go!\n") # go!

    ### STUDENT CODE STARTS HERE

    # We adopt different codes (algorithms) for the red and blue sides
    if team == 'red':

        """
        Red-side code
        """

        from calculateFKL import FK
        from calculateIK6L import IK
        fk=FK()

        """
        Static manipulation 
        """
        static=True
        picking_height_offset=0
        lifting_height_offset=0.02

        # Get the relative pose from the camera to the end effector
        H_ee_camera = detector.get_H_ee_camera()

        # Set the scanning configuration
        q1 = np.array([-0.2,0.05,0,-pi/2.3,0,0.05+pi/2.3,pi/4])
        # Set the stacking configuration and estimate its pose using FK
        q2 = np.array([-0.2,0.1,0,-pi/2.5,0,0.1+pi/2.5,pi/4]) + np.array([pi/7,0.1,0,0,0,0.1,0])
        p2, T2=fk.forward(np.array(q2))

        # Initiate the stacking height
        max_h=T2[2,-1]
        T2[2,-1]-=0.345

        # Move robot to the scanning configuration
        arm.safe_move_to_position(q1)
        # Compute the pose of the scanning configuration using FK
        p, T=fk.forward(arm.get_positions())
        # Get the end-effector-to-camera pose
        H_ee_camera = detector.get_H_ee_camera()

        print("End effector to camera pose:")
        print(H_ee_camera)

        # Set some initial transformation matrices
        T_cam_to_end=np.array([[0,1,0,0],[-1,0,0,0.05],[0,0,1,0],[0,0,0,1]])
        T_end_to_cam = np.linalg.inv(T_cam_to_end)
        T_obj_to_end=np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])

        # Count for static manipulation iterations
        count=0
        # Open the gripper
        arm.exec_gripper_cmd(0.099, 1)

        # Detect and go through all the static blocks
        for (name, pose) in detector.get_detections():
            print(name,'\n',pose)
            # Get the block pose in camera frame
            Pose = np.array(pose)
            if static == False:
                break #(uncomment this if you want to try dynamics only)

            # Adjust and process the pose of the block to ensure that it can be reached by the robot arm
            if (np.abs(Pose[2,1])>0.93):
                Pose[:3,:3]=np.concatenate((Pose[:3,2].reshape(3,1),Pose[:3,0].reshape(3,1),Pose[:3,1].reshape(3,1)),axis=1)
            if (np.abs(Pose[2,0])>0.93):
                Pose[:3,:3]=np.concatenate((Pose[:3,1].reshape(3,1),Pose[:3,2].reshape(3,1),Pose[:3,0].reshape(3,1)),axis=1)
            print("pose:")
            print(Pose)

            # Compute the block pose in end-effector frame
            pose_e = H_ee_camera @ Pose
            if (pose_e[2,2]<0):
                pose_e = pose_e @ T_obj_to_end

            # Get the least-energy orientation to grab the block
            norm_T=1000
            Pose_e = np.array(pose_e)
            for pose_i in range(4):
                pose_e = pose_e @ np.array([[0,-1,0,0],[1,0,0,0],[0,0,1,0],[0,0,0,1]])
                if np.linalg.norm(pose_e-np.eye(4)) < norm_T:
                    Pose_e = np.array(pose_e)
                    norm_T = np.linalg.norm(pose_e-np.eye(4))
            print("Object pose w.r.t. end effector frame:")
            print(Pose_e)

            # Compute the block pose in robot base frame
            T_=T @ Pose_e
            print("Object pose w.r.t. world frame:")
            print(T_)

            # Compute the pre-picking pose which is 0.1m above the block and estimate its configuration using IK
            T_[2,-1]+=0.1
            for j in range(2):
                target={'R':T_[:3,:3], 't': T_[:3,-1].reshape(3)}
                q_sol=IK().panda_ik(target)
                if len(q_sol)==0:
                    T_ = T_ @ np.array([[0,1,0,0],[-1,0,0,0],[0,0,1,0],[0,0,0,1]])
                else:
                    break
            print(q_sol)

            # Always choose the least-energy pose (IK solution)
            norm_min=100
            q_now=arm.get_positions()
            for i in range(len(q_sol)):
                if np.linalg.norm(q_sol[i,:]-q_now.reshape(7))<norm_min:
                    norm_min=np.linalg.norm(q_sol[i,:]-q_now)
                    q_sol_=q_sol[i,:].reshape(7)

            # Move to the pre-picking pose
            arm.safe_move_to_position(q_sol_)

            # Compute the picking pose and estimate its configuration using IK
            T_[2,-1]-=0.103+picking_height_offset
            target1={'R':T_[:3,:3], 't': T_[:3,-1].reshape(3)}
            q_sol1=IK().panda_ik(target1)
            print(q_sol1)

            # Always choose the least-energy pose (IK solution)
            norm_min=100
            q_now=arm.get_positions()
            for i in range(len(q_sol1)):
                if np.linalg.norm(q_sol1[i,:]-q_now.reshape(7))<norm_min:
                    norm_min=np.linalg.norm(q_sol1[i,:]-q_now)
                    q_sol_1=q_sol1[i,:].reshape(7)
            # Move to the picking pose
            arm.safe_move_to_position(q_sol_1)
            # Close the gripper to grab the block
            arm.exec_gripper_cmd(0.02, 75)

            # Compute the block-lifting pose (about 0.1m above the block) and estimate its configuration using IK
            T_[2,-1]=T2[2,-1]+0.1+lifting_height_offset
            target2={'R':T_[:3,:3], 't': T_[:3,-1].reshape(3)}
            q_sol2=IK().panda_ik(target2)
            norm_min=100
            q_now=arm.get_positions()
            for i in range(len(q_sol2)):
                if np.linalg.norm(q_sol2[i,:]-q_now.reshape(7))<norm_min:
                    norm_min=np.linalg.norm(q_sol2[i,:]-q_now)
                    q_sol_2=q_sol2[i,:].reshape(7)
            # Lift the block to ensure motion safety and collision-free
            arm.safe_move_to_position(q_sol_2)

            # Compute the pre-stacking pose and estimate its configuration using IK
            target3int={'R':T2[:3,:3], 't': T2[:3,-1].reshape(3)+np.array([0,0,0.12])}
            q_placeint=IK().panda_ik(target3int)
            # Always choose the least-energy pose (IK solution)
            norm_min=100
            q_now=arm.get_positions()
            for i in range(len(q_placeint)):
                if np.linalg.norm(q_placeint[i,:]-q_now.reshape(7))<norm_min:
                    norm_min=np.linalg.norm(q_placeint[i,:]-q_now)
                    q_place_int=q_placeint[i,:].reshape(7)
            # Move to the pre-stacking pose
            arm.safe_move_to_position(q_place_int)

            # Update the stacking height, compute the stacking pose and estimate its configuration using IK
            T2[2,-1]+=0.05
            target3={'R':T2[:3,:3], 't': T2[:3,-1].reshape(3)}
            q_place=IK().panda_ik(target3)

            # Always choose the least-energy pose (IK solution)
            norm_min=100
            q_now=arm.get_positions()
            for i in range(len(q_place)):
                if np.linalg.norm(q_place[i,:]-q_now.reshape(7))<norm_min:
                    norm_min=np.linalg.norm(q_place[i,:]-q_now)
                    q_place_=q_place[i,:].reshape(7)

            # Move to the stacking pose
            arm.safe_move_to_position(q_place_)
            # Open the gripper to release the block
            arm.exec_gripper_cmd(0.09, 1)
            # Count in this round
            count+=1

            # Move to the intermediate pose and prepare for the next block
            arm.safe_move_to_position(q2-np.array([0,0,0,0.1,0,0,0]))

        """
        This function is the "Ambusher" dynamic manipulation method.
        :param hardware: boolean, whether on hardware or not
        :param arm_speed: speed of arm, 0.2 by default (not used anymore, please ignore it)
        :param max_itr_of_grabbing: maximum iterations of gripping
        :param aim_num_block: how many blocks do you want to grab
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
        Dynamic_Manipulation_Red(hardware=True, arm_speed=0.2, max_itr_of_grabbing=10, aim_num_block=5, robot_to_table_dis=0.99, min_gripper_pos=0.04, grip_time=1, height_offset=0.2, time_of_sleep=11, time_of_wait=9, picking_height_offset=-0.01, dynamics_x_offset=0.0, dynamics_y_offset=0.005, dynamics_z_offset=0.00)
        ### You only need to adjust the input parameters

    else:

        """
        Blue-side code
        """

        import calculateFK
        import calculateIK6
        from time import sleep

        fk = calculateFK.FK()
        ik = calculateIK6.IK()

        ### STUDENT CODE HERE

        start_time = time_in_seconds()
        # Define the goal area (stacking pose)
        goal_pose = np.zeros((4, 4))
        goal_pose[0:3, 0:3] = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
        goal_pose[3, 3] = 1

        if team == 'blue':
            goal_pose[0:3, 3] = np.array([0.56, -0.16, 0.230])
        else:
            goal_pose[0:3, 3] = np.array([0.61, 0.16, 0.230])

        grab_time = []
        height = 0
        arm.exec_gripper_cmd(idle_grip_width, force)

        # TESTING
        # Just for testing purpose! Please ignore it.
        testing = False
        if testing:
            dynamic_start = np.zeros((4, 4))
            if team == 'blue':
                dynamic_start[0:3, 0:3] = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
                dynamic_start[0:3, 3] = np.array([0, -0.75, 0.4])
            else:
                dynamic_start[0:3, 0:3] = np.array([[-1,0,0],[0,1,0],[0,0,-1]])
                dynamic_start[0:3, 3] = np.array([0, 0.75, 0.4])
            dynamic_start[3, 3] = 1
            q_dynamic = ik.find_grasp_IKconfig(arm, dynamic_start, enforce=True)
            q_dynamic[4] += 0.1
            q_dynamic[6] += 0.1
            end_T = np.zeros((4, 4))
            end_T[0, 3] = 0
            end_T[1, 3] = 0.75
            end_T[2, 3] = 0.224
            # end_T[1, 3] = -0.44
            # end_T[2, 3] = 0.77
            #A = np.array([[0,-1,0],[0,0,-1],[1,0,0]])
            #B = np.array([[0.71,0,-0.71],[0,1,0],[0.71,0,0.71]])
            A = np.array([[0,1,0],[0,0,1],[1,0,0]])
            B = np.array([[0.71,0,-0.71],[0,1,0],[0.71,0,0.71]])
            end_T[0:3, 0:3] = A @ B
            end_T[3, 3] = 1

            # q = start_position.copy()
            # q[0] -= pi/2
            # _, T = fk.forward(q)
            # print(T)
            # print(end_T)
            # q = ik.find_grasp_IKconfig(arm, end_T, enforce=True)
            # arm.safe_move_to_position(q)
            # _, T = fk.forward(q)
            # print(T)
            # quit()

            arm.safe_move_to_position(q_dynamic)

            for height in range(20):
                #end_T[1, 3] -= 0.01
                end_T[1, 3] += 0.01
                print(end_T)
                # Move to the pose above goal
                q_opt = ik.find_grasp_IKconfig(arm, end_T, enforce=True)
                if len(q_opt) == 0:
                    print('OPT FAIL')
                    continue
                ootw_T = end_T.copy()
                ootw_T[2, 3] += 0.1
                print(ootw_T)
                q_ootw = ik.find_grasp_IKconfig(arm, ootw_T, enforce=True)
                if len(q_ootw) == 0:
                    print('OOTW FAIL')
                    continue
                new_goal_pose = goal_pose.copy()
                new_goal_pose[0:3, 0:3] = A @ B
                new_goal_pose[2, 3] += height * 0.05
                new_goal_pose[2, 3] += 0.01
                ootw_place_pose = new_goal_pose.copy()
                ootw_place_pose[2, 3] += 0.1

                print(ootw_place_pose)

                q_ootw_place = ik.find_grasp_IKconfig(arm, ootw_place_pose, enforce=True)

                #arm.safe_move_to_position(q_ootw)
                #arm.safe_move_to_position(q_opt)
                arm.safe_move_to_position(q_ootw_place)
                #arm.safe_move_to_position(q_dynamic)
            quit()

            for height in range(1,10):
                new_goal_pose = goal_pose.copy()
                new_goal_pose[0:3, 0:3] = np.array([[0,-1,0],[0,0,-1],[1,0,0]])
                new_goal_pose[2, 3] += height * 0.05
                ootw_place_pose = new_goal_pose.copy()
                ootw_place_pose[2, 3] += 0.1

                print(ootw_place_pose)

                # Move to the pose above goal
                q_ootw_place = ik.find_grasp_IKconfig(arm, ootw_place_pose, enforce=True)
                #q_ootw_place = ik.find_grasp_IKconfig(arm, ootw_place_pose)
                arm.safe_move_to_position(q_dynamic)
                arm.safe_move_to_position(q_ootw_place) # on your mark!

        """
        Static Manipulation
        """
        if team == 'blue':
            static_start_position = np.array([0.35, -0.76012354,  0.01978261, -2.34205014, 0, 1.54119353+pi/2-1.1, 0.75344866])
        else:
            static_start_position = np.array([-0.35, -0.76012354,  0.01978261, -2.34205014, 0, 1.54119353+pi/2-1.1, 0.75344866])

        # Move to start configuration and get all the poses and names of the static blocks detected by the vision system
        arm.safe_move_to_position(static_start_position)
        name, T0b, pos_blocks = fk.scan_blocks()

        # For each detected static block, pick and place them
        for i in range(len(name)):
            print(name[i])

            # Move to the pre-picking pose above block and prepare to pick it:
            T_block = T0b[i].copy()
            # Set the pre-picking pose as 0.1m above the center of the block
            T_block[2, 3] += 0.1
            # Compute its config using IK
            q_ootw_pick = ik.find_grasp_IKconfig(arm, T_block)
            # Call singularity() function to prevent weird results if singularity appears
            if q_ootw_pick[1] == 0:
                q_ootw_pick = singularity(team, q_ootw_pick, T_block)
            arm.safe_move_to_position(q_ootw_pick)

            # Move down to reach the block (at picking pose) and pick it:
            T_block[2, 3] -= 0.1
            # Compute its configuration using IK
            q_opt = ik.find_grasp_IKconfig(arm, T_block)
            # Call singularity() function to prevent weird results if singularity appears
            if q_opt[1] == 0:
                q_opt = singularity(team, q_opt, T_block)
            arm.safe_move_to_position(q_opt)

            # Grab the block and move up back to the pre-picking pose
            grip_width = block_width - 0.01
            # Close the gripper to grip the block
            arm.exec_gripper_cmd(grip_width, force)
            arm.safe_move_to_position(q_ootw_pick)

            # Compute the goal pose for stacking and the pre-stacking pose (0.1m above the goal)
            new_goal_pose = goal_pose.copy()
            new_goal_pose[2, 3] += height * 0.05
            #if q_opt[1] == 0:
            #    new_goal_pose[0, 3] += 0.01
            ootw_place_pose = new_goal_pose.copy()
            ootw_place_pose[2, 3] += 0.1

            # Move to the pre-stacking pose (0.1m above the goal):
            # Compute its configuration using IK
            q_ootw_place = ik.find_grasp_IKconfig(arm, ootw_place_pose)
            arm.safe_move_to_position(q_ootw_place)

            # Move down and release the block:
            # Compute its configuration using IK
            q_goal = ik.find_grasp_IKconfig(arm, new_goal_pose)
            arm.safe_move_to_position(q_goal)
            # Update the height
            height += 1

            # Open the gripper to release the block:
            arm.exec_gripper_cmd(idle_grip_width, force)
            arm.safe_move_to_position(q_ootw_place)
            #arm.safe_move_to_position(arm.neutral_position())

        # Show the static manipulation time
        static_end_time = time_in_seconds()
        print('total static time:', static_end_time - start_time)

        """
        Dynamic Manipulation (Predator)
        """
        # Initiate the start (scanning) pose and some hard-code poses
        dynamic_start = np.zeros((4, 4))
        if team == 'blue':
            dynamic_start[0:3, 0:3] = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
            dynamic_start[0:3, 3] = np.array([0, -dynamic_observe_y, dynamic_observe_z])
            A = np.array([[0,-1,0],[0,0,-1],[1,0,0]])
            B = np.array([[0.71,0,-0.71],[0,1,0],[0.71,0,0.71]])
        else:
            dynamic_start[0:3, 0:3] = np.array([[-1,0,0],[0,1,0],[0,0,-1]])
            dynamic_start[0:3, 3] = np.array([0, dynamic_observe_y, dynamic_observe_z])
            A = np.array([[0,1,0],[0,0,1],[1,0,0]])
            B = np.array([[0.71,0,-0.71],[0,1,0],[0.71,0,0.71]])
        dynamic_start[3, 3] = 1
        q_dynamic = ik.find_grasp_IKconfig(arm, dynamic_start, enforce=True)
        q_dynamic[4] += 0.1
        q_dynamic[5] += 0.1

        # Initiate the counter for success and failure cases
        fail = 0
        success = 0
        #height = 4

        # As long as the stacked blocks are fewer than 12 in total, go on picking the dynamic blocks
        while height < 12:
            # Move to scanning pose and open the gripper
            arm.safe_move_to_position(q_dynamic)
            arm.exec_gripper_cmd(idle_grip_width, force)
            # Initiate the algorithm-related parameters
            dynamic_detected = False
            method = 1
            name_block = 'None'
            # Keep scanning and detecting dynamic blocks continuously until an "easy" one is found
            while not dynamic_detected:
                # Obtain all the blocks detected by the vision system
                name, T0b, pos_blocks = fk.scan_blocks()
                # Check if there is block detected actually
                if len(name) > 0:
                    # Get the farthest block among all the detected blocks
                    farthest_ind = get_farthest_block(team, name_block, name, pos_blocks)
                    print(name)
                    print(pos_blocks)
                    name_block = name[farthest_ind]
                    print(name_block)
                    # Get the x-axis value of the farthest block
                    farthest_x = pos_blocks[farthest_ind, 0]

                    # Estimate the pose and radius of this block w.r.t. the turntable
                    end_T, radius = dynamic_prediction(team, T0b[farthest_ind], 0, 0)

                    # Compute the y-axis value of the farthest block
                    if team == 'red':
                        farthest_y = turntable_offset - pos_blocks[farthest_ind, 1]
                    else:
                        farthest_y = -turntable_offset - pos_blocks[farthest_ind, 1]

                    # Estimate the theta (central angle) of the block
                    farthest_theta = np.arctan(abs(farthest_x/farthest_y))
                    # Check if the theta (central angle) is too small
                    if farthest_theta < 0.45:
                        name_block = 'None'
                    # Check if theta (central angle) is qualified (within certain expected range)
                    if (0.45 < farthest_theta < 0.57):
                        print('fartest_theta: ', farthest_theta)
                    # Check if this target block (e.g. position, theta) is eligible for picking
                    if ((team == 'red' and farthest_x < 0) or (team == 'blue' and farthest_x > 0)) and (radian_threshold - 0.015 < farthest_theta < radian_threshold):
                        # If qualified, continue to pick it
                        print('DETECTED!')
                        dynamic_detected = True
                        # First move to the pre-picking pose that is 0.1m higher than the block
                        above_T = end_T.copy()
                        above_T[2, 3] += 0.1

                        # Compute the pre-picking configuration using IK
                        q_ootw_pick = ik.find_grasp_IKconfig(arm, above_T, enforce=True)

                        # Compute the picking pose's configuration using IK
                        q_opt = ik.find_grasp_IKconfig(arm, end_T, enforce=True)

                        # Check if IK solution has solution
                        if (len(q_opt) == 0) or (len(q_ootw_pick) == 0):
                            # If at least one of them has no IK solution, then skip this round
                            end_T[0:3, 0:3] = A @ B
                            above_T[0:3, 0:3] = A @ B
                            # Compute their configurations using IK
                            q_ootw_pick = ik.find_grasp_IKconfig(arm, above_T, enforce=True)
                            q_opt = ik.find_grasp_IKconfig(arm, end_T, enforce=True)
                            print('NO IK for method 1, trying method 2')
                            if (len(q_opt) == 0) or (len(q_ootw_pick) == 0):
                                print('NO IK for method 2, skip')
                                # Skip this round to detect new blocks
                                continue
                            else:
                                if enable_hard:
                                    method = 2
                                    sleep(sleep_hard)
                                else:
                                    # Skip this round to detect new blocks
                                    continue
                        print('REACHABLE!')
                        print('METHOD', method)

                        # Move to pre-picking pose
                        arm.safe_move_to_position(q_ootw_pick)
                        # Move to picking pose
                        arm.safe_move_to_position(q_opt)

                        # Close the gripper to grab the block and move up
                        grip_width = block_width - 0.03
                        arm.exec_gripper_cmd(grip_width, force)
                        #if team == 'blue':
                        #    if (len(q_ootw_pick != 0)):
                        #       arm.safe_move_to_position(q_ootw_pick)

                        # Check if gripping is successful by calling check_grip()
                        grip_success = check_grip(arm, gripper_threshold)

                        # If gripping is failed, skip this round
                        if not grip_success:
                            fail += 1
                            print("FAIL # ", fail)
                            continue

                        # If gripping is successful, set the new goal (stacking) pose and the pre-stacking pose
                        new_goal_pose = goal_pose.copy()
                        if team == 'red':
                            new_goal_pose[0:3, 0:3] = np.array([[0,1,0],[0,0,1],[1,0,0]])
                        else:
                            new_goal_pose[0:3, 0:3] = np.array([[0,-1,0],[0,0,-1],[1,0,0]])
                        if method == 2:
                            new_goal_pose[0:3, 0:3] = A @ B
                        # Update the related poses and heights
                        new_goal_pose[2, 3] += height * 0.05
                        ootw_place_pose = new_goal_pose.copy()
                        new_goal_pose[2, 3] += 0.005
                        ootw_place_pose[2, 3] += 0.1

                        # Compute the configuration of the pre-stacking pose that is above the goal (stacking pose)
                        q_ootw_place = ik.find_grasp_IKconfig(arm, ootw_place_pose, enforce=True)

                        # Check if the pre-stacking pose has IK solution
                        if len(q_ootw_place) == 0 and method == 2:
                            C = np.array([[0.71,0,0.71],[0,1,0],[-0.71,0,0.71]])
                            new_goal_pose[0:3, 0:3] = A @ C
                            ootw_place_pose = new_goal_pose.copy()
                            ootw_place_pose[2, 3] += 0.095
                            q_ootw_place = ik.find_grasp_IKconfig(arm, ootw_place_pose, enforce=True)

                        # Move to the pre-stacking pose
                        arm.safe_move_to_position(q_ootw_place)

                        # Check if the gripping is successful by calling check_grip()
                        grip_success = check_grip(arm, gripper_threshold)
                        # If yes, add 1 to success counter; if not, add 1 to failure counter and skip this round
                        if grip_success:
                            success += 1
                            print("SUCCESS # ", success)
                        else:
                            fail += 1
                            print("FAIL # ", fail)
                            continue

                        # Compute the configuration of stacking pose using IK
                        q_goal = ik.find_grasp_IKconfig(arm, new_goal_pose, enforce=True)
                        # Move down to the stacking pose
                        arm.safe_move_to_position(q_goal)
                        # Update the height parameter
                        height += 1
                        # Open the gripper to release the block
                        arm.exec_gripper_cmd(idle_grip_width, force)
                        # Move back to the pre-stacking pose
                        arm.safe_move_to_position(q_ootw_place)

                        # Record and show the dynamic manipulation time for this round
                        end_grabbing = time_in_seconds()
                        print('total dynamic time:', end_grabbing - static_end_time)
                        print('total time:', end_grabbing - start_time)
                sleep(0.1)


        # Record and print out all the relevant data for dynamic manipulation
        end_time = time_in_seconds()
        dynamic_time = end_time - static_end_time
        static_time = static_end_time - start_time
        print("Static time: ", static_time)
        print("Dynamic time: ", dynamic_time)
        print("Total time: ", static_time + dynamic_time)
        print("Success rate:", success / (fail+success))




