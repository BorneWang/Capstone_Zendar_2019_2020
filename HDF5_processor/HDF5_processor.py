#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 10:28:08 2019

@author: bowenwang
"""

import argparse
import utils

def main():
    parser = argparse.ArgumentParser(description='HDF5 data processor',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--src', type=str, default='ffbp_output_good.h5', help='HDF5 file')
    parser.add_argument('--task', type=str, help='processor task')
    parser.add_argument('--video', type=bool, default=False, help='generate video or not')
    parser.add_argument('--dir', type=str, help='dir save images')
    args = parser.parse_args()

    #Load source file
    f = utils.Load_src(args.src)
    
    #Generate npy matrix cascade
    if args.task == 'npy':
        utils.generate_npy(f,args.dir)
        
    if args.task == 'png':
        utils.generate_png(f,args.dir)
        #if args.video == True:
            #utils.generate_video(args.dir)
    
    
    
if __name__ == '__main__':
    main()