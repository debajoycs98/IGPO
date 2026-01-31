import yaml
import json
import os
from tqdm import tqdm
import time
import concurrent.futures
import traceback
import glob
from util import FileSystemReader, string_to_uuid
from pathlib import Path
import datetime
from loguru import logger
import random
    
class TaskScheduler:
    def __init__(self, agent_config):  
        self.agent_config = agent_config
        self.data_writing_path = self.agent_config["data_writing_path"]
        if self.data_writing_path[-1] == '/':
            self.data_writing_path = self.data_writing_path[:-1]
        self.fs_reader = FileSystemReader(
            oss_access_key_id = agent_config['oss_access_key_id'],
            oss_access_key_secret = agent_config['oss_access_key_secret'],
            oss_endpoint = agent_config['oss_endpoint']
        )
        self.work_states = {} 
        self.task_dict = {} #key: consumer name value: task_list
        self.task_idx_dict = {} #key: consumer name value: idx2index 找回每个idx的位置（idx之间可能有空档）
        self.task_type_consumer_dict = {} #key: work_type 
        self.task_type_worker_dict = {} #key: worker name 
        self.cache_dict = {} #key: work_type value:cache
        self.logger = logger

        
    def find_active_workers(self):
        folders = self.fs_reader.list_files(self.data_writing_path)['directories']
        self.work_dict = {}
        for filepath in [x for x in folders]:
            try:
                if filepath == 'cache':
                    continue
                if not self.fs_reader.exists(self.data_writing_path + '/' + filepath + "/heartbeat"):
                    self.logger.info(f"注销不活跃的节点{filepath}")
                    self.fs_reader.delete_file(self.data_writing_path + '/' + filepath + '/')
                    continue 
                heartbeat = json.loads(self.fs_reader.read_file(self.data_writing_path + '/' + filepath + "/heartbeat"))
                target_time = datetime.datetime.strptime(heartbeat['time'], "%Y%m%d_%H%M%S")
                current_time = datetime.datetime.now()
                time_diff = abs((current_time - target_time).total_seconds()) 
                if time_diff < 60:  # 60秒内活跃
                    if 'worker_' in filepath:
                        for work_type in heartbeat['work_type']: 
                            self.work_dict[work_type] = self.work_dict.get(work_type, [])
                            self.work_dict[work_type].append(filepath)
                        if filepath not in self.work_states:
                            self.work_states[filepath] = 0
                else:
                    self.logger.info(f"注销不活跃的节点{filepath}")
                    self.fs_reader.delete_file(self.data_writing_path + '/' + filepath + '/')
                    if filepath in self.work_states:
                        del self.work_states[filepath]
                    if filepath in self.task_type_worker_dict:
                        del self.task_type_worker_dict[filepath]
                    # if filepath in self.task_dict:
                    #     del self.task_dict[filepath]
                    for work_type,l in self.work_dict.items():
                        self.work_dict[work_type] = [x for x in self.work_dict[work_type] if x != filepath]
            except Exception as e:
                self.logger.error(f"find_active_workers {filepath}: {str(traceback.format_exc())}")
                return False
        self.logger.info(f"所有注册worker：{self.work_dict} 活跃状态：{self.work_states}")
    
    def process_workpath(self, workpath) -> bool:
        """处理单个workpath的任务文件，返回是否全部完成"""
        trytime = 0
        while(True):
            trytime += 1
            try:
                data_file = f"{self.data_writing_path}/{workpath}/data.json"
                task_list = json.loads(self.fs_reader.read_file(data_file))
                all_finished = True
                i = 0
                for content in task_list:
                    if content.get('finish', 0) != 1:
                        all_finished = False
                    else:
                        if 'idx' not in content:
                            print('没有找到idx',content)
                            content['idx'] = i
                        content['workpath'] = workpath
                        consumer_path = content['consumerpath']
                        if consumer_path not in self.task_idx_dict:
                            print('----consumer_path已死',consumer_path)
                            continue
                        if content['idx'] not in self.task_idx_dict[consumer_path]:
                            print('----异常',content)
                            continue
                        pos = self.task_idx_dict[consumer_path][content['idx']]
                        if consumer_path in self.task_dict:
                            self.task_dict[consumer_path][pos] = content
                            if pos == 0:
                                print(consumer_path,consumer_path in self.task_dict,content)
                                print(self.task_dict[consumer_path][pos])

                            # 不知为什么，两个字典里的content没有同步改变
                            # work_type = content['tool_call']['name']
                            # pos2 = self.task_type_consumer_idx_dict[consumer_path][content['idx']]
                            # self.task_type_consumer_dict[work_type][pos2] = content
                    i += 1
                
                if all_finished or trytime == 20:
                    self.logger.info(f"{workpath} 有 {len(task_list)} 个任务，{'全部完成' if all_finished else '有待处理'}")
                    return all_finished
            except Exception as e:
                self.logger.error(f"{workpath} 结果读取错误: {str(traceback.format_exc())}")
                return False
            time.sleep(3)
        
    def run_loop(self):
        # 先找所有需要处理的任务和消费者
        self.task_dict = {}
        self.task_idx_dict = {}
        self.task_type_consumer_dict = {}
        self.task_type_worker_dict = {}
        folders = self.fs_reader.list_files(self.data_writing_path)['directories']
        for filepath in [x for x in folders if 'consumer_' in x]:
            data_file = self.data_writing_path + '/' + filepath + "/data.json"
            if self.fs_reader.exists(data_file):
                task_list = json.loads(self.fs_reader.read_file(data_file))  
                isfinish = False
                pos = 0
                for content in task_list:
                    self.task_idx_dict[filepath] = self.task_idx_dict.get(filepath, {})
                    self.task_idx_dict[filepath][content['idx']] = pos
                    pos += 1
                    if content.get('finish', 0) != 0:
                        isfinish = True
                    else:
                        try:
                            content['consumerpath'] = filepath
                        except Exception as e:
                            content['content'] = f"输入错误: {str(traceback.format_exc())}"
                            content['finish'] = 1
                if not isfinish and len(task_list) > 0:
                    self.task_dict[filepath] = task_list
                    print('=====',filepath, len(task_list))

            
        if len(self.task_dict) == 0:
            self.logger.info('暂时没有要处理的任务。')
            time.sleep(3)
            return
            
        remain_task = 0
        for filepath, l in self.task_dict.items():
            for i in range(len(l)):
                content = l[i]
                if content.get('finish', 0) == 0:
                    remain_task += 1
        if remain_task == 0:
            self.logger.info("所有任务完成")
            time.sleep(5)
            return
        self.logger.info(f'收到{remain_task}个新任务。')

        self.task_type_worker_dict = {}
        all_done = {}
        while(True):
            self.task_type_consumer_dict = {}
            for filepath, task_list in self.task_dict.items():
                for content in task_list:
                    if content.get('finish', 0) == 0:
                        try:
                            work_type = content["tool_call"]["name"]
                            content['consumerpath'] = filepath
                            self.task_type_consumer_dict[work_type] = self.task_type_consumer_dict.get(work_type, [])
                            self.task_type_consumer_dict[work_type].append(content)
                        except Exception as e:
                            content['content'] = f"输入错误: {str(traceback.format_exc())}"
                            content['finish'] = 1
                        
            # 找到所有worker
            self.find_active_workers()
            
            # 为所有任务分配worker
            for work_type, l in self.task_type_consumer_dict.items():
                l_unfinish = [x for x in l if x.get('finish', 0) == 0]
                if random.random() > 0.5:
                    l_unfinish = l_unfinish[::-1] #随机倒序，负载均衡
                self.logger.info(f'{work_type}有{len(l_unfinish)}个任务待处理')
                for i in range(len(l_unfinish)):
                    content = l_unfinish[i]
                    worker_valid = [x for x in self.work_dict.get(work_type,[]) if self.work_states[x] == 0]
                    if content.get('finish', 0) == 1:
                        # 任务已经处理好了
                        continue
                    elif len(self.work_dict.get(work_type,[])) == 0:
                        self.logger.info(f'No {work_type} worker active!')
                        content['content'] = f'No {work_type} worker active!'
                        content['finish'] = 1
                    else:
                        # 均匀分配任务 同一个任务可能下发两遍
                        if len(worker_valid) == 0:
                            continue
                        worker_idx = i % len(worker_valid)
                        workpath = worker_valid[worker_idx]
                        self.task_type_worker_dict[workpath] = self.task_type_worker_dict.get(workpath, [])
                        if len(self.task_type_worker_dict[workpath]) < max(len(l)//4,100): #  一次只下发50个同类任务
                            self.task_type_worker_dict[workpath].append(content)

            # 给有空的机器下发任务
            for workpath, l in self.task_type_worker_dict.items():
                self.logger.info(f"{workpath}状态:{'空闲中' if self.work_states.get(workpath, 0) == 0 else '工作中'}")
                if len(l) > 0:
                    data_file = self.data_writing_path + '/' + workpath + "/data.json"
                    self.fs_reader.write_file(data_file, json.dumps(l, indent=4, ensure_ascii=False))
                    self.logger.info(f"{workpath} 接收到'{len(l)}':个任务")
                    self.work_states[workpath] = 1
                    
            # """并发检查所有workpath的任务状态"""
            all_done = {}
            if len(self.task_type_worker_dict) > 0:
                with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                    futures = {}
                    # 提交所有workpath的检查任务
                    for workpath, l in self.task_type_worker_dict.items():
                        if len(l) > 0:
                            futures[executor.submit(self.process_workpath, workpath)] = workpath
                    # 检查每个任务的结果
                    for future in concurrent.futures.as_completed(futures):
                        workpath = futures[future]
                        all_done[workpath] = future.result()
                        if all_done[workpath]:
                            self.work_states[workpath] = 0
                            del self.task_type_worker_dict[workpath]
                    self.logger.info(f"所有任务处理结果：{all_done}" )
                
                
            remain_task = 0
            for filepath, l in self.task_dict.items():
                for i in range(len(l)):
                    content = l[i]
                    if content.get('finish', 0) == 0:
                        remain_task += 1
            if remain_task == 0 and all([v for k,v in all_done.items()]):
                self.logger.info("所有任务完成")
                break
            self.logger.info(f'还有{remain_task}个任务没有完成。worker状态{all_done}')
            time.sleep(3)

    
        for filepath, l in self.task_dict.items():
            if len(l) > 0:
                # 保存的时候做完的部分和没做完的任务
                self.logger.info(f"{filepath}的{len(l)}个任务交付。")
                data_file = self.data_writing_path + '/' + filepath + "/data.json"
                self.fs_reader.write_file(data_file, json.dumps(l, indent=4, ensure_ascii=False))
    
    def run(self):
        self.logger.info(f"{self.data_writing_path}注册中心启动")
        while True:
            self.run_loop()

            

if __name__ == "__main__":
    config = yaml.safe_load(open("./tools_server/config.yaml"))    
    scheduler = TaskScheduler(agent_config=config)
    scheduler.run()
