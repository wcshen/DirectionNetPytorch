# 线段s->e 与 圆(圆心c, 半径r)的交点
# https://longjin666.cn/?p=793

import math
from time import time
import numpy as np
from pip import main

def project(p1, p2, c):
    """
    p1, p2, c: (2,) ndarray [x, y]
    """
    base = p2 - p1
    base_norm_2 = np.sum(base**2)
    c_p1 = c - p1
    r = np.dot(base, c_p1) / base_norm_2  # double
    
    pr = p1 + base*r
    
    return pr

def within_p1_p2(crosspoint, p1, p2):
    if (p1[0] <= crosspoint[0] <= p2[0]) and (p1[1] <= crosspoint[1] <= p2[1]):
        return True
    return False

def get_crosspoint(p1, p2, c, r):
    pr = project(p1, p2, c)
    base = math.sqrt(r**2 - np.sum((pr - c)**2))
    p2_p1_length = np.linalg.norm(p2-p1, ord=2)
    e = (p2-p1) / p2_p1_length
    crosspoint1 = pr + e * base
    crosspoint2 = pr - e * base
    if within_p1_p2(crosspoint1, p1, p2):
        return crosspoint1
    else:
        return crosspoint2
    
    
def demo():
    p1 = np.array([0.99, 0.99])
    p2 = np.array([1.01, 1.01])
    c = np.array([0, 0])
    r = 1.0
    cross_point = get_crosspoint(p2, p1, c , r)
    print(cross_point)
    theta = math.atan2(-cross_point[1], cross_point[0])
    theta = math.degrees(theta)
    print(f"theta = {theta:.2f}")

    
if __name__ == '__main__':
    import time
    t1 = time.time()
    demo()
    t2 = time.time()
    cost_time = (t2-t1) * 1000.0
    print(f"cost_time = {cost_time:.2f} ms")