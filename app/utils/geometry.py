import numpy as np

def euclidean(p1, p2):
    return np.linalg.norm(p1 - p2)

def calculate_ear(eye):
    p1,p2,p3,p4,p5,p6 = eye[0],eye[1],eye[2],eye[3],eye[4],eye[5]
    ear = (euclidean(p2,p6) + euclidean(p3,p5))/(2*euclidean(p1,p4))
    return ear

def calculate_mar(mouth):
    p1,p2,p3,p4,p5,p6 = mouth[0],mouth[1],mouth[2],mouth[3],mouth[4],mouth[5]

    mar = (euclidean(p2,p6) + euclidean(p3,5))/(2*euclidean(p1,p4))
    return mar




