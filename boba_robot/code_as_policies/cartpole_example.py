from boba_robot.code_as_policies.lmp_core import lmp_fgen

prompt_f_gen = """
import numpy as np
import math
from shapely.geometry import *
from shapely.affinity import *
from utils import get_obj_outer_pts_np

# define function: total = get_total(xs=numbers).
def get_total(xs):
    return np.sum(xs)

# define function: y = eval_line(x, slope, y_intercept=0).
def eval_line(x, slope, y_intercept):
    return x * slope + y_intercept

# define function: pt_np = get_pt_to_the_left(pt_np, dist).
def get_pt_to_the_left(pt_np, dist):
    delta = np.array([-dist, 0])
    return translate_pt_np(pt_np, delta=delta)

# define function: pt_np = get_pt_to_the_top(pt_np, dist).
def get_pt_to_the_top(pt_np, dist):
    delta = np.array([0, dist])
    return translate_pt_np(pt_np, delta=delta)

# define function line = make_line_by_length(length=x).
def make_line_by_length(length):
  start = np.array([0, 0])
  end = np.array([length, 0])
  line = make_line(start=start, end=end)
  return line

# define function: line = make_vertical_line_by_length(length=x).
def make_vertical_line_by_length(length):
  line = make_line_by_length(length)
  vertical_line = rotate(line, 90)
  return vertical_line

# define function: pt = interpolate_line(line, t=0.5).
def interpolate_line(line, t):
  pt = line.interpolate(t, normalized=True)
  return np.array(pt.coords[0])

# example: scale a line by 2 around the centroid.
line = make_line_by_length(1)
new_shape = scale(line, xfact=2, yfact=2, origin='centroid')

# example: rotate a point around origin by 45 degrees.
pt = Point([1,1])
new_pt = rotate(pt, 45, origin=[0, 0])

# example: getting object points of object0.
pts_np = get_obj_outer_pts_np('object0')
""".strip()

prompt_f_gen_exec = """
import numpy as np
import math
from shapely.geometry import *
from shapely.affinity import *

# define function: total = get_total(xs=numbers).
def get_total(xs):
    return np.sum(xs)

# define function: y = eval_line(x, slope, y_intercept=0).
def eval_line(x, slope, y_intercept):
    return x * slope + y_intercept

# define function: pt = get_pt_to_the_left(pt, dist).
def get_pt_to_the_left(pt, dist):
    return pt + [-dist, 0]

# define function: pt = get_pt_to_the_top(pt, dist).
def get_pt_to_the_top(pt, dist):
    return pt + [0, dist]

# define function line = make_line_by_length(length=x).
def make_line_by_length(length):
  line = LineString([[0, 0], [length, 0]])
  return line

# define function: line = make_vertical_line_by_length(length=x).
def make_vertical_line_by_length(length):
  line = make_line_by_length(length)
  vertical_line = rotate(line, 90)
  return vertical_line

# define function: pt = interpolate_line(line, t=0.5).
def interpolate_line(line, t):
  pt = line.interpolate(t, normalized=True)
  return np.array(pt.coords[0])
""".strip()


def main():
    context_vars = {}
    exec(prompt_f_gen_exec, context_vars)

    f_name = "keep_pole_upright_with_pd_control"
    f_sig = "direction = keep_pole_upright_with_pd_control(x, x_dot, theta, theta_dot)"
    info = "direction is 1 if going right, 0 if going left"

    f_name = "find_average_np"
    f_sig = "av = find_average_np(1, 2)"
    info = "use numpy functions"

    policy = lmp_fgen(
        prompt_f_gen, f_name, f_sig, recurse=True, context_vars=context_vars, info=info
    )
    print(policy)
    #  print(policy(1, 2, 3, 4))
    print(policy(1, 2))


if __name__ == "__main__":
    main()
