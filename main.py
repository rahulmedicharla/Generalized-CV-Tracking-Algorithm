import sys
from descriptor import generate_covariance_matrix
from tracking import covariance_tracking

def main(target_img: str, source_file: str, type: str, debug: bool):
    # go across different version types and run corresponding one
    if type == "V1":
        cov_matrix, dimensions = generate_covariance_matrix(target_img, debug)
        covariance_tracking(source_file, cov_matrix, dimensions)
        
if __name__ == "__main__":
    args = sys.argv

    if "-target" not in args or "-source" not in args or "-type" not in args:
        print("Please run program like python main.py -target <target_img_path> -source <source_path> -type<algorithm version> --debug?")
    else:
        main(args[2], args[4], args[6], '--debug' in args)