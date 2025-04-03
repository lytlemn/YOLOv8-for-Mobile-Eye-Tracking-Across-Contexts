import argparse
from bffs_v8 import bffs_pipe
import warnings
parser = argparse.ArgumentParser()


parser.add_argument("-s", "--subject", dest="subject", help="Subject Number")
parser.add_argument("-v", "--video", action='store_true', help="Generate Video?")

args = parser.parse_args()
subject = args.subject
vidgen = args.video

def main():
    subjects = ["","",""] #subject numbers to process many at once
    for subject in subjects:
        warnings.simplefilter("ignore")
        print("running subject " + subject)
        bffs_pipe(subject, True)

if __name__ == "__main__":
    main()
