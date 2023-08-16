#!/usr/bin/env python3
#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import sys
import argparse
import RPi.GPIO as GPIO
import time

pin_mode = GPIO.setmode(GPIO.BCM)

from abnormal.handler import AbnormalHandler

from jetson_inference import poseNet
from jetson_utils import videoSource, videoOutput, Log, cudaFont

abnormal = AbnormalHandler()

# parse the command line
parser = argparse.ArgumentParser(description="Run pose estimation DNN on a video/image stream.",
                                 formatter_class=argparse.RawTextHelpFormatter,
                                 epilog=poseNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="resnet18-body", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="links,keypoints", help="pose overlay flags (e.g. --overlay=links,keypoints)\nvalid combinations are:  'links', 'keypoints', 'boxes', 'none'")
parser.add_argument("--threshold", type=float, default=0.15, help="minimum detection threshold to use")

try:
	args = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# load the pose estimation model
net = poseNet(args.network, sys.argv, args.threshold)

# create video sources & outputs
input = videoSource(args.input, argv=sys.argv)
output = videoOutput(args.output, argv=sys.argv)

import torch
font = cudaFont()

pins = []
for i in range(41):
    try:
        GPIO.setup(i, GPIO.OUT)
        pins.append(i)
    except Exception as e:
        print(i, "\t", "N/A")

for p in pins:
    GPIO.output(p, GPIO.LOW)

frame_idx = 0
# process frames until EOS or the user exits
while True:
    # capture the next image
    img = input.Capture()
    # frame_idx += 1
    # print("Current frame is {}!!!!!!!!!!!!!!!!!!!!!!".format(frame_idx))

    if img is None: # timeout
        continue

    # perform pose estimation (with overlay)
    poses = net.Process(img, overlay=args.overlay)

    # print the pose results
    print("detected {:d} objects in image".format(len(poses)))

    for pose in poses:
        print(pose)
        print(pose.Keypoints)
        print('Links', pose.Links)

    torch_kps = []
    for pose in poses:
        kps_tmp = []
        valid_idx = [p.ID for p in pose.Keypoints]
        valid_pose = [[p.x, p.y] for p in pose.Keypoints]
        for idx in range(17):
            if idx in valid_idx:
                kps_tmp.append(valid_pose[valid_idx.index(idx)])
            else:
                kps_tmp.append([0, 0])
        torch_kps.append(kps_tmp)

    torch_kps = torch.tensor(torch_kps)
    ids = torch.ones(torch_kps.shape[0])
    boxes = torch.ones((torch_kps.shape[0], 4))
    kps_score = torch.ones((torch_kps.shape[0], 17, 1))
    abnormal.process(ids, boxes, torch_kps, kps_score)
    print(abnormal.status)

    # M 8 9 10 11
    if abnormal.status.sum() > 0:
        wordings = "Wheelchair detected"
        GPIO.output(8, GPIO.LOW)
        GPIO.output(9, GPIO.HIGH)
        time.sleep(0.05)
    else:
        wordings = "Normal"
        GPIO.output(9, GPIO.LOW)
        GPIO.output(8, GPIO.HIGH)
        time.sleep(0.05)

    # print(wordings)
    font.OverlayText(img, img.width, img.height, "{:s} ".format(wordings), 5, 5, font.White)

    # render the image
    output.Render(img)

    # update the title bar
    output.SetStatus("{:s} | Network {:.0f} FPS | {:s}".format(args.network, net.GetNetworkFPS(), wordings))

    # print out performance info
    net.PrintProfilerTimes()

    # exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        for p in pins:
            GPIO.output(p, GPIO.LOW)
        break
