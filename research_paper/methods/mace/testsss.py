import numpy as np

def mixed_distance(x, y, discrete, continuous, class_name, ddist, cdist):
    xd = [x[att] for att in discrete if att != class_name]
    wd = 0.0
    dd = 0.0
    if len(xd) > 0:
        yd = [y[att] for att in discrete if att != class_name]
        wd = 1.0 * len(discrete) / (len(discrete) + len(continuous))
        dd = ddist(xd, yd)

    xc = np.array([x[att] for att in continuous])
    wc = 0.0
    cd = 0.0
    if len(xd) > 0:
        yc = np.array([y[att] for att in continuous])
        wc = 1.0 * len(continuous) / (len(discrete) + len(continuous))
        cd = cdist(xc, yc)

    return wd * dd + wc * cd

def distance_function(x0, x1, discrete, continuous, class_name):
    return mixed_distance(x0, x1, discrete, continuous, class_name,
                          ddist=simple_match_distance,
                          cdist=normalized_euclidean_distance)


def normalized_euclidean_distance(x, y):
    return 0.5 * np.var(x - y) / (np.var(x) + np.var(y))


def simple_match_distance(x, y):
    count = 0
    for xi, yi in zip(x, y):
        if xi == yi:
            count += 1
    sim_ratio = 1.0 * count / len(x)
    return 1.0 - sim_ratio


x = {'x2': 1.0, 'x3': 1.0, 'x0': 1.0, 'x1': 2.0, 'x4': 2.0}
y = {'x2': 1.0, 'x3': 1.0, 'x0': 1.0, 'x1': 2.0, 'x4': 2.0}
discrete = ['x2', 'x1', 'x4']
continuous = ['x0', 'x3']
class_name = 'y'

dd = distance_function(x, y, discrete, continuous, class_name)
print(dd)