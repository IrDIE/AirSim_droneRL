import numpy as np
import airsim

# Define the initial positions of teh environments being used.
# The module name is the name of the environment to be used in the config file
# and should be same as the .exe file (and folder) within the unreal_envs folder

def indoor_meta():
    orig_ip = [     #x, y, theta in DEGREES

                    # One - Pyramid
                    [-21593, -1563, -45],  # Player Start
                    [-22059, -2617, -45],
                    [-22800, -3489, 90],

                    # Two - FrogEyes
                    [-15744, -1679, 0],
                    [-15539, -3043, 180],
                    [-13792, -3371, 90],

                    # Three - UpDown
                    [-11221, -3171, 180],
                    [-9962, -3193, 0],
                    [-7464, -4558, 90],

                    # Four - Long
                    [-649, -4287, 180],  # Player Start
                    [-4224, -2601, 180],
                    [1180, -2153, -90],

                    # Five - VanLeer
                    [6400, -4731, 90],  # Player Start
                    [5992, -2736, 180],
                    [8143, -2835, -90],

                    # Six - Complex_Indoor
                    [11320, -2948, 0],
                    [12546, -3415, -180],
                    [10809, -2106, 0],

                    # Seven - Techno
                    [19081, -8867, 0],
                    [17348, -3864, -120],
                    [20895, -4757, 30],

                    # Eight - GT
                    [26042, -4336, 180],
                    [26668, -3070, 0],
                    [27873, -2792, -135]



                ]# x, y, theta
    level_name = [
                    'Pyramid1', 'Pyramid2', 'Pyramid3',
                    'FrogEyes1', 'FrogEyes2', 'FrogEyes3',
                    'UpDown1', 'UpDown2', 'UpDown3',
                    'Long1', 'Long2', 'Long3',
                    'VanLeer1', 'VanLeer2', 'VanLeer3',
                    'ComplexIndoor1', 'ComplexIndoor2', 'ComplexIndoor3',
                    'Techno1', 'Techno2', 'Techno3',
                    'GT1', 'GT2', 'GT3',
                ]
    crash_threshold = 0.07
    return orig_ip, level_name, crash_threshold


# Train complex indoor initial positions
def indoor_complex():
    # The environment can be downloaded from
    # https://drive.google.com/drive/u/2/folders/1u5teth6l4JW2IXAkZAg1CbDGR6zE-v6Z
    orig_ip = [
        [-195, 812, 0],  # Player start
        [-1018, 216, -50],
        [-77, -118, 180],
        [800, -553, 190]
    ]
    level_name = ['Complex1', 'Complex2', 'Complex3', 'Complex4']
    crash_threshold = 0.07
    return orig_ip, level_name, crash_threshold

# Test condo indoor initial positions


def outdoor_courtyard(get_far_distance = False):

    # coordinates for .reset() , Format: [x coord, y coord, yaw degrees] in picture coordinates that can be directly be passed to airsim Pose class
    orig_ip = [
        [0, 0, 0],  # same as simulation starts
        [-0.57, 16, 90],
        [4.83, 15.66, 0],
        [22.33, 14.37, -90],
        [26.36,-10.5, 135],
        [-11.97, -8.53, 0]
    ]
    done_xy = [(-23,37),(-32,34)] # min_x, max_x min_y, max_y
    #crash_threshold = 0.07
    return orig_ip, done_xy

def indoor_maze_easy():
    orig_ip = [
        [0, 0, 90],  # same as simulation starts

    ]
    done_xy = [(-5,5),(-7,60)] # min_x, max_x min_y, max_y
    # crash_threshold = 0.07
    return orig_ip, done_xy







def get_airsim_position(name):
    """
    :return:
        airsim_positions_raw - need for initial json generation and correct start of exe environment
        airsim_positions - list of initial positions on map of environment (see retrive_initial_position.py)

    """
    airsim_positions_raw = [0, 0, 0, 0, 0, 0]
    name = name+'()'
    initial_positions, done_xy = eval(name)
    return initial_positions, airsim_positions_raw, done_xy
