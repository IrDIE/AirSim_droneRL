import airsim 
import os
possible_host = os.environ.get('REAL_HOST_WSL', default = "127.0.0.1")

try: 
    client = airsim.MultirotorClient() 
    client.confirmConnection()
except:
    from loguru import logger as logg 
    logg.info(f"retry with ip = {possible_host} ")
    del client
    client = airsim.MultirotorClient(ip=possible_host) 
    client.confirmConnection()
