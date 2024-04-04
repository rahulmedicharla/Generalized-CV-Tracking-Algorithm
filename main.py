import sys

def main(target_img, source_file, type, debug):
    pass

if __name__ == "__main__":
    args = sys.argv

    if "-target" not in args or "-source" not in args or "-type" not in args:
        print("Please run program like python main.py -target <target_img_path> -source <source_video_path> -type<descriptor & tracking algorithm type> --debug?")
    else:
        main(args[2], args[4], args[6], '--debug' in args)