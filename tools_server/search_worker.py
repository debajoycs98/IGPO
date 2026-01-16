import yaml
from types import SimpleNamespace
import json
import os
import concurrent.futures
from tqdm import tqdm
import time
import threading
import traceback
import glob
from openai import OpenAI
from tools_server.util import MessageClient
from tools_server.tools import Handler



if __name__ == "__main__":
    config = yaml.safe_load(open("./tools_server/config.yaml"))
    handler = Handler(agent_config=config)
    client = MessageClient(config['data_writing_path'],
                           isconsumer = False,
                           oss_access_key_id=config['oss_access_key_id'],
                           oss_access_key_secret=config['oss_access_key_secret'],
                           oss_endpoint=config['oss_endpoint'],
                           handler = handler, 
                           work_type = config['available_tools'],
                          )
    client.maintain_worker()