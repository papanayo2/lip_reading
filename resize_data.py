import cv2
import numpy as np
import moviepy.editor as mpy
import moviepy
import os, sys
from os import listdir
from os.path import isfile, join
import time


def make_resized_samples(directory, newdirectory, new_size):
    for video in listdir(directory):
        clip = mpy.VideoFileClip(join(directory, video)).resize(new_size)
        clip.write_videofile(join(newdirectory, video))

start_time = time.time()
current_directory = os.path.dirname(os.path.abspath(__file__))

path1 = join(current_directory, 'videos', 'video1')
path2 = join(current_directory, 'videos', 'video2')
path3 = join(current_directory, 'videos', 'video3')
path5 = join(current_directory, 'videos', 'video5')

paths = [path1, path2, path3, path5]


sizes = np.arange(0.9, 0.3, -0.05)
for j in range(len(paths)):
    for new_size in sizes:
        directory_name = 'video%d_%f' % (j+1, new_size)
        new_directory = join(current_directory, 'videos', directory_name)
        os.mkdir(new_directory)
        make_resized_samples(paths[j], new_directory, new_size)
    
print("--- %s seconds ---" % (time.time() - start_time))    


    