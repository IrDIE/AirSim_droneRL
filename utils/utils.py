import os, json
from configparser import ConfigParser

import pandas as pd
from dotmap import DotMap
import numpy as np
import cv2

def generate_json(cfg, initial_positions):
    flag  = True
    path = os.path.expanduser('~\Documents\Airsim')
    if not os.path.exists(path):
        os.makedirs(path)

    filename = path + '\settings.json'

    data = {}

    if cfg.mode == 'move_around':
        data['SimMode'] = 'ComputerVision'
    else:
        data['SettingsVersion'] = 1.2
        data['LocalHostIp'] = cfg.ip_address
        data['SimMode'] = cfg.SimMode
        data['ClockSpeed'] = cfg.ClockSpeed
        data["ViewMode"]= "NoDisplay"
        PawnPaths = {}
        PawnPaths["DefaultQuadrotor"] = {}
        PawnPaths["DefaultQuadrotor"]['PawnBP'] = ''' Class'/AirSim/Blueprints/BP_''' + cfg.drone + '''.BP_''' + cfg.drone + '''_C' '''
        data['PawnPaths']=PawnPaths


        Vehicles = {}

        for agents in range(cfg.num_agents):
            name_agent = "drone" + str(agents)
            agent_position = initial_positions
            Vehicles[name_agent] = {}
            Vehicles[name_agent]["VehicleType"] = "SimpleFlight" # PhysXCar, SimpleFlight, PX4Multirotor, ComputerVision, ArduCopter & ArduRover
            Vehicles[name_agent]["X"] = agent_position[0]
            Vehicles[name_agent]["Y"] = agent_position[1]
            Vehicles[name_agent]["Z"] = agent_position[2]
            #Vehicles[name_agent]["Z"] = 0
            Vehicles[name_agent]["Yaw"] = agent_position[3]
        data["Vehicles"] = Vehicles

        CameraDefaults = {}
        CameraDefaults['CaptureSettings']=[]
        # CaptureSettings=[]

        camera = {}
        camera['ImageType'] = 0
        camera['Width'] = cfg.width
        camera['Height'] = cfg.height
        camera['FOV_Degrees'] = cfg.fov_degrees

        CameraDefaults['CaptureSettings'].append(camera)

        camera = {}
        camera['ImageType'] = 3
        camera['Width'] = cfg.width
        camera['Height'] = cfg.height
        camera['FOV_Degrees'] = cfg.fov_degrees

        CameraDefaults['CaptureSettings'].append(camera)

        data['CameraDefaults'] = CameraDefaults
    with open(filename, 'w') as outfile:
        json.dump(data, outfile, indent=4)

    return flag

def ConvertIfStringIsInt(input_string):
    try:
        float(input_string)

        try:
            if int(input_string) == float(input_string):
                return int(input_string)
            else:
                return float(input_string)
        except ValueError:
            return float(input_string)

    except ValueError:
        true_array = ['True', 'TRUE', 'true', 'Yes', 'YES', 'yes']
        false_array = ['False', 'FALSE', 'false', 'No', 'NO', 'no']
        if input_string in true_array:
            input_string = True
        elif input_string in false_array:
            input_string = False

        return input_string


def visualize_observation(observation, k = 2):
    frames3 = observation[0].get_frames()
    if k == 3:
        fr1, fr2, fr3 = frames3[:3, :, :].transpose(1, 2, 0), frames3[3:6, :, :].transpose(1, 2, 0), frames3[6:9, :,
                                                                                                 :].transpose(1, 2, 0)
        fr = np.concatenate((fr1, fr2, fr3), axis=1)
        while True:
            observation_scaled = cv2.resize(fr, (fr.shape[1], fr.shape[0]))
            cv2.imshow('', observation_scaled)
            if cv2.waitKey(33) == ord('q'): break
    if k == 2:
        fr1, fr2 = frames3[:3, :, :].transpose(1, 2, 0), frames3[3:6, :, :].transpose(1, 2, 0)
        fr = np.concatenate((fr1, fr2), axis=1)
        while True:
            observation_scaled = cv2.resize(fr, (fr.shape[1], fr.shape[0]))
            cv2.imshow('', observation_scaled)
            if cv2.waitKey(33) == ord('q'): break
            if cv2.waitKey(33) == ord('p'): return 0


def create_folder(SAVE_PATH):
    os.makedirs(name=SAVE_PATH, exist_ok=True)

from loguru import logger
def update_logg_reward(df : pd.DataFrame = None, restart_n = 0, reward = -1, duration = 0):

    df = df.append({
        'restart_n' : restart_n,
        'reward' : reward,
        'duration' : duration
               },ignore_index=True)

    return df

def load_save_logg_reward(csv_rewards_log , save_path, df = None,save = True):
    if save : df.to_csv(os.path.join(save_path, '..', f'{csv_rewards_log}.csv'))
    else: # load
        try:
            return pd.read_csv(os.path.join(save_path, '..', f'{csv_rewards_log}.csv'))
        except:
            # no reward logs exist. return empty df
            return pd.DataFrame(columns=['restart_n', 'reward', 'duration'])

def read_cfg(config_filename='configs/main.cfg', verbose=False):
    parser = ConfigParser()
    parser.optionxform = str
    parser.read(config_filename)
    cfg = DotMap()

    if verbose:
        hyphens = '-' * int((80 - len(config_filename))/2)
        print(hyphens + ' ' + config_filename + ' ' + hyphens)

    for section_name in parser.sections():
        if verbose:
            print('[' + section_name + ']')
        for name, value in parser.items(section_name):
            value = ConvertIfStringIsInt(value)
            cfg[name] = value
            spaces = ' ' * (30 - len(name))
            if verbose:
                print(name + ':' + spaces + str(cfg[name]))

    return cfg

