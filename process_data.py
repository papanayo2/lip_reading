import os
from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
import dlib
from scipy import stats
import tensorflow as tf
import sys
import math

def draw_face(image, x, y, w, h):
    """
        Draws a rectangle around face.
    """
    cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0), 2)

def draw_lip_points(image, coords, border_x, border_y):
    i=0
    for (x, y) in coords:
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
        text = '%d'%(i)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, text, (x,y), font, 0.3, (0,255,0), 1, cv2.LINE_AA) 
        i+=1
        
def create_images(frame_array, frame_count):
    """
        Creates image files out of the processed frames.
    """
    for i in range(frame_count):
        cv2.imwrite('frame%d.jpg' % i, frame_array[:,:,i])
                   


def extract_mouth_points(shape, dtype='int'):
    """
        Extracts the facial landmarks around the mouth.
    """
    mouth_points = np.zeros((20, 2), dtype=dtype)
    j = 0
    for i in range(48, 68):
        mouth_points[j] = (shape.part(i).x, shape.part(i).y)
        j += 1
    return mouth_points



def average(lis):
    return sum(lis)/len(lis)

def remove_false_detections(face_detections, frame_dims):
    """
        Removes detections that are not about a face. If the
        size of the frame is less half the size of the average face rectangle 
        size then remove it. 
    """
    for face in face_detections:
        (x, y, w, h) = face
        if w*h < (average(frame_dims))*0.5:
            face_detections.remove(face)
        else:
            frame_dims.append(w*h) # add size to list to further calculate the average
            
    return (face_detections, frame_dims)

def dist(point1, point2):
    """
        Calculates the distance between two points.
    """
    (x1, y1) = point1
    (x2, y2) = point2
    a = x2-x1
    b = y2-y1
    if a == 0 & b==0:
        return 0
    else:
        return math.sqrt((a)**2 + (b)**2)

def calculate_distances(array):
    """
        Calculates the distance between the specified points.
    """
    new_array = []
    new_array.append(dist(array[1],array[11]))
    new_array.append(dist(array[2],array[10]))
    new_array.append(dist(array[3],array[9]))
    new_array.append(dist(array[4],array[8]))
    new_array.append(dist(array[5],array[7]))
    new_array.append(dist(array[0],array[6]))
    new_array.append(dist(array[12],array[16]))
    new_array.append(dist(array[13],array[19]))
    new_array.append(dist(array[14],array[18]))
    new_array.append(dist(array[15],array[17]))
    return np.array(new_array)
    
def get_videos_from_directory(path):
    videos = []
    for directory in listdir(path):
        dire = join(path, directory)
        for file in listdir(dire):
            videos.append(join(dire, file))
        
    return videos


def get_features(path_name):

    if path_name == 'testing_set' or path_name == 'training_set':
        current_directory = os.path.dirname(os.path.abspath(__file__))
        
        path = join(current_directory, 'testing_set')
        
        videos = get_videos_from_directory(path)
    else:
         videos = [path_name]   
    #declare 68 point predictor
    #pre-trained data
    data = 'shape_predictor_68_face_landmarks.dat'
    predictor = dlib.shape_predictor(data)
    
    #declare face detector
    #pre-trained haar cascade xml file from opencv & intel.
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    #stores features of all videos in dataset
    features = np.zeros((len(videos), 10), dtype=np.uint8)
    
    j = 0 
    for video in videos:
        #open video
        cap = cv2.VideoCapture(video)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        #declare arrays
        #store - stores frames
        frames = np.zeros((height, width, frame_count), dtype=np.uint8)
        #points - stores the features of every frame
        points = np.zeros((frame_count, 10), dtype=np.uint8)
        
        frame_dims = [0] # stores face detection width*height of rectangle in order to calculate average. 
        i = 0
    
        while(cap.isOpened()):
        
            ret, frame = cap.read()
        
            if ret == False: #if no more frames
                break
            #convert to frame to grayscale.
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # detect faces in frame.
            faces = face_cascade.detectMultiScale(gray, 1.1, 6)
           
            #have to deal with situation where it detects two or more faces
            if len(faces) > 0:
                #print(faces)
                faces = faces.tolist()
                # remove false detections
                (faces, frame_dims) = remove_false_detections(faces, frame_dims)
                        
                if len(faces) > 0:    
                    (x, y, w, h) = faces[0]
             
                    #border_x = x
                    #border_y = y
            
                    #draw_face(gray, x, y, w, h)
                        
                    shape = predictor(gray, dlib.rectangle(x, y, x+w, y+h))#applies points within rectangle
                    shape = extract_mouth_points(shape)
                    distances = calculate_distances(shape)
                    #if row is all zeros then don't keep it.
                    if np.count_nonzero(distances) == 0:
                        continue
                    else:
                        points[i, :] = distances
                        
                        #draw_lip_points(gray, shape, border_x, border_y)
                        #cv2.imshow('ghg',gray)
    
            if len(faces)>0:    
                frames[:,:,i] = gray # this line is causing problems. It saves useless frames.
                i+=1
                
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
                
                
    
        points = points[0:i, :] #remove left over zeros elements
          
        p = np.std(points, 0) #apply standard deviation across matrix
        p = np.round_(p)
    
        features[j, :] = p 
        j+=1
        #create_images(frames, points.shape[0])
            
       
        cap.release()
        cv2.destroyAllWindows()
    #save matrices
    #np.save('data.npy', features)
    #np.savetxt('data.txt', features)
    return features


np.set_printoptions(threshold=np.nan)
#get_features('training_set')
#get_features('testing_set')