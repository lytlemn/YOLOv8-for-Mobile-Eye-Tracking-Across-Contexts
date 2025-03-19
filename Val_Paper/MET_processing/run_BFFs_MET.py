import argparse
from bffs_v8 import bffs_pipe
import warnings
parser = argparse.ArgumentParser()

# -sub SUBJECT -vid VIDEO
parser.add_argument("-s", "--subject", dest="subject", help="Subject Number")
parser.add_argument("-v", "--video", action='store_true', help="Generate Video?")

args = parser.parse_args()
subject = args.subject
vidgen = args.video

def main():
    #If processing on file at a time uncomment below and comment out the many at once section
    # global subject, vidgen, picgen
    # message = ("Proceed with processing data for subject " + str(subject) + " [y/n] \n")
    # if input(message) == "y":
    #     warnings.simplefilter("ignore")
    #     bffs_pipe(subject, vidgen)
    subjects = ["","",""] #subject numbers to process many at once
    for subject in subjects:
        warnings.simplefilter("ignore")
        print("running subject " + subject)
        bffs_pipe(subject, True)

if __name__ == "__main__":
    main()
