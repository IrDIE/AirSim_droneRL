# Apply Reinforcement Learning in AirSim simulation (with drone) :star:

This repo contains several training alghorithms for trainig drone avoid obstackles in AirSim simulation

### What is AirSim?
official repo : 

https://github.com/microsoft/AirSim

AirSim good at simulation of drone physic and render of environment good enough:


| <img src="https://github.com/IrDIE/AirSim_droneRL/blob/main/readme_pictures/airsim_drone_.gif" width="640" height="353"/>                |
|------------------------------------------------------------------------------------------------------------------------------------------|
| <img src="https://media.giphy.com/media/vFKqnCdLPNOKc/giphy.gif" width="20" height="20"/>  most important - AirSim have nice API :heart: |





AirSim environment are wrapped in Gym environment (check `airsim_env.py`) so we can interact with it jist like with any Gym environment.

* To launch environment you should have .exe file with rendered UnrealEngine environment (and UE4 installed as well). 
  * You can find zipped .exe environments here - https://github.com/microsoft/AirSim/releases
  * Also setup instruction to make AirSim API work can be found in official AirSim repo

<details>
  <summary>Why use AirSim?</summary>
  AirSim have ArduPilot and ROS support - what can be very helpful if you gonna do inference in real-world

  <img src="https://github.com/IrDIE/AirSim_droneRL/blob/main/readme_pictures/why_airsim.png" width="705" height="408"/>  
  source - https://imrclab.github.io/workshop-uav-sims-icra2023/papers/RS4UAVs_paper_10.pdf


</details>


