import unittest

import numpy as np
import math
import vis
import cv2
import argparse
import sys
import os

import vis
from gym_duckietown.envs import DuckietownEnv


def find_intersection(lines):
    x01 = lines[0]
    y01 = lines[1]
    x11 = lines[2]
    y11 = lines[3]
    x02 = 0
    y02 = 240
    x12 = 640
    y12 = 240
    a1 = y11 - y01
    a2 = y12 - y02
    b1 = x01 - x11
    b2 = x02 - x12
    c1 = -x01 * y11 + y01 * x11
    c2 = -x02 * y12 + y02 * x12
    x = (b1 * c2 - b2 * c1) / (b2 * a1 - b1 * a2)
    y = (a2 * c1 - a1 * c2) / (a1 * b2 - a2 * b1)
    if 0 <= x <= 640:
        return True
    else:
        return False


def calculate_angle(lines):
    first = lines[0]
    second = lines[1]
    x1 = first[0][1] - first[0][0]
    x2 = second[0][0] - second[0][1]
    y1 = first[0][3] - first[0][2]
    y2 = second[0][2] - second[0][3]
    a = np.array([x1, y1])
    b = np.array([x2, y2])
    # print(x1, y1, "-", x2, y2)
    # print(np.linalg.norm(a, ord=2), "-", np.linalg.norm(b, ord=2))

    scalar = np.dot(a, b)
    angle = scalar / (np.linalg.norm(a, ord=2) * np.linalg.norm(b, ord=2))
    return math.degrees(math.acos(angle))


lines_list = []


class TestMarkup(unittest.TestCase):

    def setUp(self):
        if len(lines_list) == 0:
            image_count = 50
            env = DuckietownEnv(
                seed=5123123,  # random seed
                map_name="udem1",
                max_steps=100,  # we don't want the gym to reset itself
                camera_width=640,
                camera_height=480,
                full_transparency=True,
                distortion=True,
                domain_rand=False
            )
            for i in range(image_count):
                action = (1.0, 1.0)  # your agent here (this takes random actions)
                observation, reward, done, info = env.step(action)
                lane_lines = vis.detect_lane(observation)
                lines_list.append(lane_lines)
                if done:
                    observation = env.reset()

    def test_empty_lines(self):
        for i in range(len(lines_list)):
            with self.subTest(i=i):
                self.assertIsNotNone(lines_list[i], "Line is empty")

    def test_count_lines(self):
        for i in range(len(lines_list)):
            with self.subTest(i=i):
                self.assertEqual(len(lines_list[i]), 2, "Should be 2 lines")

    def test_lines_angle(self):
        for i in range(len(lines_list)):
            with self.subTest(i=i):
                if len(lines_list[i]) != 2:
                    self.assertEqual(len(lines_list[i]), 2, "Should be 2 lines")
                else:
                    angle = calculate_angle(lines_list[i])
                    self.assertTrue(0 <= angle <= 30, "Invalid angle between lines")

    def test_lines_position(self):
        for i in range(len(lines_list)):
            with self.subTest(i=i):
                if len(lines_list[i]) != 2:
                    self.assertEqual(len(lines_list[i]), 2, "Should be 2 lines")
                else:
                    left_line = lines_list[i][0][0]
                    right_line = lines_list[i][1][0]
                    self.assertTrue(find_intersection(left_line) & find_intersection(right_line),
                                    "Less than 2 lines drawn")


if __name__ == '__main__':
    unittest.main()
