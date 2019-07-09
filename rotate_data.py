import os
import numpy as np
import moviepy.editor as mpy
import moviepy
from os import listdir
from os.path import isfile, join
import time


def make_rotated_samples(directory, newdirectory, rotation):
    for video in listdir(directory):
        clip = mpy.VideoFileClip(join(directory, video)).rotate(rotation)
        clip.write_videofile(join(newdirectory, video))

start_time = time.time()
current_directory = os.path.dirname(os.path.abspath(__file__))

path = join(current_directory, 'training_set')

for directory_name in listdir(path):
    new_directory_name = '%sr5' % (directory_name)
    new_directory = join(path, new_directory_name)
    if not os.path.exists(new_directory):
        os.mkdir(new_directory)
    directory = join(path, directory_name)
    make_rotated_samples(directory, new_directory, 5)
    
print("--- %s seconds ---" % (time.time() - start_time))    


    