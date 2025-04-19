from agents.dueling_double_dqn_PER import train_DDDQN_per, infer_DDDQN_per
from agents.ddpg_per import train_DDPG_per, infer_DDPG_per
from agents.ddpg_per import Actor

def setup(cfg):
    return CFG_NAME_MAPPING[cfg]


CFG_NAME_MAPPING = {
    # "configs/agents_conf/dueling_double_dqn.ini": [train_DDDQN, infer_DDDQN],
    "configs/agents_conf/dueling_double_dqn_PER.ini": [
        train_DDDQN_per,
        infer_DDDQN_per,
    ],
    # "configs/agents_conf/ddpg.ini": [train_DDPG, infer_DDPG],
    "configs/agents_conf/ddpg_per.ini": [train_DDPG_per, infer_DDPG_per],
}
