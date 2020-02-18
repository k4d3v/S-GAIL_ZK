import numpy as np


def targets(target_pose, dist = 0.025):
    # Check if the target is within some bounds
    pos = 0.15
    t00 = np.linalg.norm(target_pose-[-pos, -pos]) < dist
    t01 = np.linalg.norm(target_pose-[-pos, pos]) < dist
    t10 = np.linalg.norm(target_pose-[pos, -pos]) < dist
    t11 = np.linalg.norm(target_pose-[pos, pos]) < dist
    
    return t00, t01, t10, t11

def goal_reached(target_pose, finger_pose):
    # Look if goal was reached
    return 1 if np.linalg.norm(
        list(map(lambda x: 0.15 if x > 0 else -0.15, target_pose)) - finger_pose) < 0.025 else 0
