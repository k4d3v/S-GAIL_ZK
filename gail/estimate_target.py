import numpy as np


def targets(target_pose, dist = 0.025):
    pos = 0.15
    t00 = np.linalg.norm(target_pose-[-pos, -pos]) < dist
    t01 = np.linalg.norm(target_pose-[-pos, pos]) < dist
    t10 = np.linalg.norm(target_pose-[pos, -pos]) < dist
    t11 = np.linalg.norm(target_pose-[pos, pos]) < dist
    
    return t00, t01, t10, t11
