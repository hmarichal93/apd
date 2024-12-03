import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2
import pandas as pd
from pathlib import Path
import time
import argparse

from automatic_wood_pith_detector.image import resize_image_using_pil_lib, Color
from automatic_wood_pith_detector.automatic_wood_pith_detector import  apd, apd_pcl, apd_dl, Method


def main(filename, output_dir, percent_lo=0.5, st_w=3, method=0, new_shape=640, debug=True, lo_w=11, st_sigma=1.2,
        weigths_path=None):

    to = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    import os
    os.system(f"rm -rf {output_dir}/*")
    # 1.0 load image
    img_in = cv2.imread(filename)
    o_height, o_width = img_in.shape[:2]
    # 1.1 resize image
    if new_shape > 0:
        img_in = resize_image_using_pil_lib(img_in, height_output=new_shape, width_output=new_shape)


    if debug:
        cv2.imwrite(str(output_dir / 'resized.png'), resize_image_using_pil_lib(img_in, 640,640))


    if method == Method.apd:
        print("apd")
        peak = apd(img_in, st_sigma, st_w, lo_w, rf = 7, percent_lo = percent_lo, max_iter = 11, epsilon =10 ** -3,
                       debug=debug, output_dir = output_dir)
    elif method == Method.apd_pcl:
        print("apd_pcl")
        peak = apd_pcl(img_in, st_sigma, st_w, lo_w, rf = 7, percent_lo = percent_lo, max_iter = 11, epsilon =10 ** -3,
                           debug=debug, output_dir = output_dir)

    elif method == Method.apd_dl:
        print("apd_dl")
        peak = apd_dl(img_in, output_dir, weigths_path)

    else:
        raise ValueError(f"method {method} not found")

    if debug:
        img = img_in.copy()
        H, W, _ = img.shape
        dot_size = H // 200
        x, y = peak
        img = cv2.circle(img,  (np.round(x).astype(int), np.round(y).astype(int)), dot_size, Color.blue, -1)
        cv2.imwrite(str(Path(output_dir) / 'peak.png'), resize_image_using_pil_lib(img, 640, 640))

    tf = time.time()
    # 3.0 save results
    new_shape_h, new_shape_w = img_in.shape[:2]
    convert_original_scale = lambda peak: (peak * np.array([o_width/new_shape_w,o_height/new_shape_h])).tolist()

    peak = convert_original_scale(peak) if new_shape>0 else peak

    data = {'coarse': np.array(peak), 'exec_time(s)':tf-to}

    df = pd.DataFrame(data)
    #print(df)
    df.to_csv(str(output_dir / 'pith.csv'), index=False)

    return peak



if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Pith detector')
    #input arguments parser.
    parser.add_argument('--filename', type=str, required=True, help='input image')
    parser.add_argument('--output_dir', type=str, required=True, help='output directory')

    #method parameters
    parser.add_argument('--new_shape', type=int, default=0, help='new shape')
    ##add pclines parameter
    parser.add_argument('--method', type=int, default=False, help='method: 0 for apd, 1 for apd_pcl, 2 for apd_dl')
    parser.add_argument('--weigths_path', type=str, default='checkpoints/yolo/all_best_yolov8.pt', help='weigths_path')
    parser.add_argument('--debug', type=bool, default=False, help='debug')



    args = parser.parse_args()


    params = dict(filename=args.filename, output_dir=args.output_dir, new_shape=args.new_shape,
                debug=args.debug, method=args.method, weigths_path=args.weigths_path)
    main(**params)


