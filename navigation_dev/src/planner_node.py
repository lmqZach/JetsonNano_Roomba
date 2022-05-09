#!/usr/bin/env python

import rospy
import cv2
import apriltag
from std_msgs.msg import Float32MultiArray
from navigation_dev.msg import AprilDetections
from navigation_dev.msg import Pose 
import numpy as np

ctrl_pub = rospy.Publisher('/ctrl_cmd', Float32MultiArray, queue_size=2)


def next(cell):
    global grid
    for dx, dy in [(0, 1), (-1, 0), (0, -1), (1, 0)]:
        nei = (cell[0]+dx, cell[1]+dy)
        if grid[nei] != 1:
            grid[nei] = 1
            return nei
    return False


def trans2waypoint(cell):
    return [0.4+cell[1]*0.2, 2.6-cell[0]*0.2]


def localize(measure):
    # given tag relative coordinate, tag world coordinate,
    # compute robot world coordinate
    robot_pos = np.zeros((2, 1))
    robot_o = []
    for tag_r_cor, tag_o, tag_pos in measure:
        robot_o_c = (tag_pos[2] - tag_o - np.pi/2) % (2*np.pi)
        tm = np.array([[np.cos(robot_o_c), -np.sin(robot_o_c)],
                       [np.sin(robot_o_c), np.cos(robot_o_c)]])
        r_trans = np.matmul(tm, tag_r_cor)
        robot_pos += np.array([tag_pos[:2]]).T - r_trans
        robot_o.append(robot_o_c)
    robot_pos /= len(measure)
    first = robot_o[0]
    if len(measure) > 1:
        o_diff = 0
        for o in robot_o[1:]:
            o_diff += o_dist(first, o)
        o_diff /= len(robot_o) - 1
        first -= o_diff
        first %= 2*np.pi
    return robot_pos, first


def o_dist(o1, o2):
    if o1 - o2 > np.pi:
        o1 -= 2*np.pi
    elif o1 - o2 < -np.pi:
        o1 += 2*np.pi
    return o1 - o2


def to_waypoint(waypoint):
    global mu, wait
    x, y, theta = mu[0][0], mu[1][0], mu[2][0]
    x_w, y_w = waypoint
    # angle is the relative angle of waypoint w.r.t. current robot location
    angle = o_dist(np.arctan2(y_w-y, x_w-x) % (2*np.pi), theta)
    dist = np.sqrt(((y_w - y) ** 2 + (x_w - x) ** 2))
    if dist < 0.1 or wait:
        wait = False
        return [0, 0]
    else:
        if abs(angle) < 0.2:
            return [0.2, 0.2]
        else:
            wait = True
            if angle > 0:
                return [0, 0.25]
            else:
                return [0.25, 0]


def pose_callback(msg):
    global mu, cell, waypoint
    cmd_msg = Float32MultiArray()
    pose_mat = np.array(msg.pose.matrix)

    # data association
    x, y, theta = mu[0][0], mu[1][0], mu[2][0]
    measure = []
    for i in range(len(pose_mat) // 4):
        feature = pose_mat[i * 4: i * 4 + 4]
        tag_id = int(feature[0])
        tag_r_cor = feature[1:3].reshape(2, 1)  # relative coordinate
        tag_o = feature[3]  # relative orientation
        tag_tm = np.array([[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta), np.cos(theta)]])
        r_trans = np.matmul(tag_tm, tag_r_cor)
        # transform to world coordinate
        tag_w_cor = r_trans + np.array([[x, y]]).T
        # compare to ground truth, get the minimum distance
        assigned_pos = -1
        min_dist = 10
        for tag_pos in TagDict[tag_id]:
            tag_w_cor_true = np.array([tag_pos[:2]]).T
            dist = np.sqrt(np.sum((tag_w_cor_true - tag_w_cor) ** 2))
            if dist < min_dist:
                min_dist = dist
                assigned_pos = tag_pos
        measure.append([tag_r_cor, tag_o, assigned_pos])

    if measure:
        robot_pos, robot_o = localize(measure)
        mu[:2] = robot_pos
        mu[2] = robot_o

    dist = np.sqrt(((waypoint[1] - mu[1][0]) ** 2 +
                    (waypoint[0] - mu[0][0]) ** 2))
    # get next waypoint
    if dist < 0.1:
        cell = next(cell)
        if cell:
            waypoint = trans2waypoint(cell)
    movement = to_waypoint(waypoint)
    print(mu, cell, movement)
    cmd_msg.data = [0.0] + movement
    ctrl_pub.publish(cmd_msg)


if __name__ == "__main__":
    wait = False
    grid = np.zeros((12, 12))
    grid[0, :] = grid[11, :] = grid[:, 0] = grid[:, 11] = 1
    cell = (1, 1)
    grid[cell] = 1
    cell = next(cell)
    waypoint = trans2waypoint(cell)
    mu = np.array([[0.6, 2.4, 0.0]]).T
    TagDict = {1: [[0.5, 0., np.pi/2], [2.5, 0, np.pi/2], [3, 1.5, np.pi],
                   [2.5, 3, 3*np.pi/2], [0.5, 3, 3*np.pi/2], [0, 1.5, 0]],
               2: [[1.5, 0, np.pi/2], [3, 0.5, np.pi], [3, 2.5, np.pi],
                   [1.5, 3, 3*np.pi/2], [0, 2.5, 0], [0, 0.5, 0]]}
    rospy.init_node('planner_node')
    rospy.Subscriber("/current_pose", Pose, pose_callback)
    rospy.spin()
