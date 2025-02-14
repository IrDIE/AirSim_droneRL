from agents import setup
from configparser import ConfigParser
from loguru import logger

def main():
    cfg_agent_path = "configs/agents_conf/ddpg_per.ini" #"configs/agents_conf/dueling_double_dqn.ini"
    cfg_env_path="configs/env_conf/cfg_NH_ddpg_per.ini"

    cfg_agent = ConfigParser()
    cfg_agent.read(cfg_agent_path)
    exe_path="unreal_envs/AirSimNH/AirSimNH/WindowsNoEditor/AirSimNH.exe"
    documents_path="../../../../../Documents"
    infer_weights = cfg_agent.get("infer", "load_from")

    train_fn, infer_fn = setup(cfg_agent_path)
    if len(infer_weights):
        logger.info(f"\n{cfg_agent_path}\n{cfg_env_path}\nStart infer ... \n")
        infer_fn(
            cfg_agent,
            exe_path=exe_path,
            cfg_env_path=cfg_env_path,
            documents_path=documents_path,
        )
    else:
        train_fn(
            cfg_agent,
            exe_path = exe_path,
            cfg_env_path = cfg_env_path,
            documents_path = documents_path,
        )

    # train_DDPG_per(logg_tb = LOGG_TB_DIR, save_path = SAVE_WEIGHTS, seq_size = 5,  height_airsim_restart_positions = [-0.8339])
    # main_dddqn()
    # inference(height_airsim_restart_positions=[-0.8339], seq_size = 5)  # main()


if __name__ == "__main__":
    main()
