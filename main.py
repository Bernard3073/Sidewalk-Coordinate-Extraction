#!/usr/bin/env python

import os
import sys
sys.path.append('/home/bernard/.local/lib/python3.7/site-packages')
import cv2

def main():
    file_dir = "/home/bernard/Akrobotix/code/picture/"
    data = [i for i in os.listdir(file_dir) if i.endswith('_pred.png')]
    print(len(data))
if __name__ == '__main__':
    main()