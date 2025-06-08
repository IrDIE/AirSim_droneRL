# Reinforcement Learning in AirSim simulation (Autonomous UAV Navigation) :star:

This repository contains several deep reinforcement learning (DRL) algorithms for training a drone to avoid obstacles in the AirSim simulation.

You can experiment with your own agents and rewards!

AirSim environment are wrapped in Gym environment so we can interact with it jist like with any Gym environment.

##### In this repo:
1. [What is AirSim?](#What-is-AirSim?)  

2. [Docker setup instructions](#Docker-setup-instructions) 

3. [Start training](#Start-training) 
<a name="headers"/>



### What is AirSim?

<details>
  <summary>Why use AirSim?</summary>

  AirSim have ArduPilot and ROS support - what can be very helpful if you gonna do inference in real-world

  <img src="https://github.com/IrDIE/AirSim_droneRL/blob/main/readme_pictures/why_airsim.png" width="705" height="408"/>  
  
  source - https://imrclab.github.io/workshop-uav-sims-icra2023/papers/RS4UAVs_paper_10.pdf

</details>

official repo : 

https://github.com/microsoft/AirSim

AirSim good at simulation of drone physic and render of environment good enough:


| <img src="https://github.com/IrDIE/AirSim_droneRL/blob/main/readme_pictures/airsim_drone_.gif" width="640" height="353"/>                |
|------------------------------------------------------------------------------------------------------------------------------------------|
| <img src="https://media.giphy.com/media/vFKqnCdLPNOKc/giphy.gif" width="20" height="20"/>  most important - AirSim have nice API :heart: |




* To launch environment you should have .exe (if you are on Windows) file with rendered UnrealEngine environment (and UE4 installed as well). 
  * You can find zipped .exe environments here - https://github.com/microsoft/AirSim/releases
  * Also setup instruction to make AirSim API work can be found in official AirSim repo

## Docker setup instructions:

This repository uses a **Docker-based setup** for training a reinforcement learning (RL) agent in Ubuntu container, while running the **Unreal Engine 4 (UE4)** simulation on the host machine.

> These instructions are based on a **Windows host**, but can be adapted for **Linux**.

---

#### 0. Strat here

- **Unreal Engine 4**: Run the AirSim environment (`.exe`) directly on the **host** machine.
- **Development Environment**: Use the `.devcontainer` feature in **VS Code** to run the development container (configured with `--net=host`).
  - More info on dev containers: [VS Code Dev Containers Documentation](https://code.visualstudio.com/docs/devcontainers/containers)


<img src="https://github.com/IrDIE/AirSim_droneRL/blob/itmo_branch/readme_pictures/docker.png" width="1720" height="300"/> 

---

#### 1. Connecting to AirSim from the Container

<details>
  <summary> <-- press spoiler</summary>

To connect from the **Docker container** to the **AirSim simulation** running on the host:

1. **AirSim Server Info**
   - Default **port**: `41451` (hardcoded in the `airsim` Python library — no need to manually configure it).
   - You’ll need the **host IP address** that the AirSim server is running on.

2. **Networking Considerations (Windows-specific)**
   - Docker on Windows runs inside **WSL**, not natively on Windows.
   - To enable proper networking between **Windows → WSL → Docker**, you need to configure WSL as follows:

</details>

---

#### 2. WSL Networking Configuration

<details>
  <summary> <-- press spoiler</summary>

1. Locate your `.wslconfig` file (typically at `C:\Users\<your-username>\.wslconfig`) and add the following line:

   ```ini
   networkingMode=mirrored
   ```
2. Open the project in VS Code using “Open Folder in Container”.

3. Inside the container, run the connection test:
    ```ini
    python3 ./test_connection_docker.py
    ```

    ✅ Success!
    If the script connects successfully — You're all set! 🎉 

    If not, please open an issue and include details about your network setup so we can help troubleshoot.

</details>

## Start training

Currently, the repository contains code for training:

  * Deep Deterministic Policy Gradient (with PER)
  * Dueling Double Deep Q Network (with PER)

To train an agent you need to set up config at `configs/agents_conf` and simply run:


```python
python3 main.py
```
