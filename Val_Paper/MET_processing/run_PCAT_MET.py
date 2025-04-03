# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:18:52 2024

@author: mnl5205
"""

import argparse
from pcat_v8 import pcat_pipe, proc_met_files
import warnings

parser = argparse.ArgumentParser()

parser.add_argument("-s", "--subject", dest="subject", help="Subject Number")
parser.add_argument("-v", "--video", action='store_true', help="Generate Video?")

args = parser.parse_args()
subject = args.subject
vidgen = args.video

def main():
        subjects = ["","",""] # Add subject numbers we want to process here
        for subject in subjects:
            warnings.simplefilter("ignore")
            print("running subject " + subject)
            try:
                gaze = proc_met_files(subject)
            except Exception as e:
                print(e)
            try:
                pcat_pipe(subject, gaze, True, True)
            except Exception as e:
                print(e)

if __name__ == "__main__":
    main()
