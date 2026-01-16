import oss2
from urllib.parse import urlparse
import os
import socket
import datetime
import time
import json
import traceback
import uuid
import threading
import json
import concurrent.futures
from loguru import logger
import traceback
             


def string_to_uuid(input_string):
    """将字符串转换为确定性 UUID (版本5)"""
    namespace = uuid.NAMESPACE_DNS  
    return uuid.uuid5(namespace, str(input_string))
    
def get_network_info():
    hostname = socket.gethostname()
    try:
        # 获取所有 IPv4 地址（可能包含回环）
        all_ips = socket.gethostbyname_ex(hostname)[2]
        # 过滤回环地址
        real_ips = [ip for ip in all_ips if not ip.startswith("127.")]
        return hostname + real_ips[0]
    except:
        return hostname

class MessageClient:
    def __init__(self, path = '', isconsumer = True, oss_access_key_id=None, oss_access_key_secret=None, oss_endpoint=None,
                handler = None, work_type = ['web_search', 'browse_webpage']):
        self.oss_access_key_id = oss_access_key_id
        self.oss_access_key_secret = oss_access_key_secret
        self.oss_endpoint = oss_endpoint
        self.path = path
        if self.path[-1] == '/':
            self.path = self.path[:-1]
        self.name = get_network_info() + '_' + datetime.datetime.now().strftime("%Y%m%d_%H%M")
        
        self.fs_reader = FileSystemReader(
            oss_access_key_id = oss_access_key_id,
            oss_access_key_secret = oss_access_key_secret,
            oss_endpoint = oss_endpoint
        )
        
        if isconsumer:
            self.nodepath = self.path + '/consumer_' + self.name + '/'
            self.work_type = []
        else:
            self.nodepath = self.path + '/worker_' + self.name + '/'
            self.handler = handler
            self.work_type = work_type

        self.heartbeat_thread = threading.Thread(target=self._update_heartbeat)
        self.heartbeat_thread.daemon = True
        self.heartbeat_thread.start()


    def _update_heartbeat(self):
        print(f"{self.path}  {self.name} 注册成功。")
        while(True):
            self.fs_reader.write_file(self.nodepath + "heartbeat",
                                      json.dumps({'time': datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                                       'work_type': self.work_type}, indent=4, ensure_ascii=False))
            time.sleep(30)

    def submit_tasks(self, task_list):
        data_path = self.nodepath + "data.json"
        self.fs_reader.write_file(data_path, json.dumps(task_list, indent=4, ensure_ascii=False))
        response_finish = False
        while not response_finish:
            try:
                output = json.loads(self.fs_reader.read_file(data_path))
                if all([x.get('finish', 0) == 1 for x in output]):
                    response_finish = True
                else:
                    time.sleep(3)
            except:
                time.sleep(3)            
        return output   

    def process_tasks(self):
        data_path = self.nodepath + "data.json"
        try:
            if not self.fs_reader.exists(data_path):
                return 0
            task_list = json.loads(self.fs_reader.read_file(data_path))
            task_list_notfinish = []
            idx = []
            for i in range(len(task_list)):
                if task_list[i].get('finish', 0) == 1:
                    continue
                else:
                    idx.append(i)
                    task_list_notfinish.append(task_list[i])
            if len(task_list_notfinish) == 0:
                # 没有新任务
                return 0
            print(f"接收到{len(task_list_notfinish)}个任务")
            handler_result_list = self.handler.handle_all(task_list_notfinish)
            for i in range(len(handler_result_list)):
                result = handler_result_list[i]
                if 'content' in result:
                    result['finish'] = 1
                task_list[idx[i]] = result
            self.fs_reader.write_file(data_path, json.dumps(task_list, indent=4, ensure_ascii=False))
            return 1
        except Exception as e:
            print("process_tasks发生异常:",str(traceback.print_exc()))  # 打印完整堆栈到标准错误
            return 0
        
    def maintain_worker(self):
        while(True):
            if self.process_tasks() == 0:
                # 没有任务
                time.sleep(3)
            
from functools import wraps

def oss_retry(max_retries=100, delay=60):
    """OSS操作重试装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except oss2.exceptions.OssError as e:
                    retries += 1
                    if retries > max_retries:
                        raise
                    time.sleep(delay)
        return wrapper
    return decorator
    
class FileSystemReader:
    def __init__(self, oss_access_key_id=None, oss_access_key_secret=None, oss_endpoint=None):
        self.oss_access_key_id = oss_access_key_id
        self.oss_access_key_secret = oss_access_key_secret
        self.oss_endpoint = oss_endpoint

    def list_files(self, path):
        """列出路径下的所有文件和文件夹（支持本地和OSS）"""
        if path.startswith("oss://"):
            return self._list_oss_files(path)
        else:
            return self._list_local_files(path)

    def _list_local_files(self, dir_path):
        """列出本地目录下的所有文件和文件夹"""
        try:
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"本地目录 {dir_path} 不存在")
            
            if not os.path.isdir(dir_path):
                raise NotADirectoryError(f"{dir_path} 不是目录")
                
            items = os.listdir(dir_path)
            return {
                'files': [f for f in items if os.path.isfile(os.path.join(dir_path, f))],
                'directories': [d for d in items if os.path.isdir(os.path.join(dir_path, d))]
            }
        except PermissionError:
            raise PermissionError(f"无权限访问本地目录 {dir_path}")
        except Exception as e:
            raise RuntimeError(f"列出本地文件失败: {str(e)}")

    @oss_retry()
    def _list_oss_files(self, oss_path):
        """列出OSS bucket下的所有文件和文件夹"""
        if oss_path[-1] != '/':
            oss_path += '/'
        parsed_url = self._parse_oss_path(oss_path)
        prefix = parsed_url.object_key
        
        auth = oss2.Auth(self.oss_access_key_id, self.oss_access_key_secret)
        bucket = oss2.Bucket(auth, self.oss_endpoint, parsed_url.bucket)
        
        try:
            # 获取所有对象
            files = []
            dirs = set()
            
            for obj in oss2.ObjectIterator(bucket, prefix=prefix, delimiter='/'):
                if obj.is_prefix():
                    # 这是一个目录
                    dir_name = obj.key[len(prefix):].rstrip('/')
                    if dir_name:  # 忽略空目录名
                        dirs.add(dir_name)
                else:
                    # 这是一个文件
                    file_name = obj.key[len(prefix):]
                    if file_name:  # 忽略空文件名
                        files.append(file_name)
            
            return {
                'files': files,
                'directories': sorted(list(dirs))
            }
        except oss2.exceptions.AccessDenied:
            raise PermissionError(f"无权限访问 OSS bucket {parsed_url.bucket}")
        except oss2.exceptions.OssError as e:
            raise RuntimeError(f"OSS 文件列表获取失败: {str(e)}")
            
    def read_file(self, file_path):
        """读取文件内容"""
        if file_path.startswith("oss://"):
            return self._read_from_oss(file_path)
        else:
            return self._read_from_local(file_path)

    def write_file(self, file_path, content, append=False):
        """写入文件内容"""
        if file_path.startswith("oss://"):
            return self._write_to_oss(file_path, content)
        else:
            return self._write_to_local(file_path, content, append)

    def _read_from_local(self, file_path):
        """读取本地文件）"""
        try:
            with open(file_path, 'rb') as f:
                return f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"本地文件 {file_path} 不存在")
        except Exception as e:
            raise RuntimeError(f"读取本地文件失败: {str(e)}")

    def _write_to_local(self, file_path, content, append):
        """写入本地文件（如果目录不存在则自动创建）"""
        if type(content) == str:
            content = content.encode('utf-8')
        mode = 'ab' if append else 'wb'
        try:
            # 确保目录存在
            dir_path = os.path.dirname(file_path)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
            # 写入文件
            with open(file_path, mode) as f:
                f.write(content)
            return True
        except PermissionError:
            raise PermissionError(f"无权限写入本地文件 {file_path}")
        except Exception as e:
            raise RuntimeError(f"写入本地文件失败: {str(e)}")

    def _read_from_oss(self, oss_path):
        """通过 OSS SDK 读取文件"""
        parsed_url = self._parse_oss_path(oss_path)
        return self._oss_read(parsed_url.bucket, parsed_url.object_key)

    def _write_to_oss(self, oss_path, content):
        """通过 OSS SDK 写入文件（覆盖模式）"""
        parsed_url = self._parse_oss_path(oss_path)
        return self._oss_write(parsed_url.bucket, parsed_url.object_key, content)

    def _parse_oss_path(self, oss_path):
        """解析 OSS 路径（oss://bucket/object-key）"""
        parsed = urlparse(oss_path)
        if not parsed.scheme == "oss":
            raise ValueError("OSS 路径格式错误，应为 oss://<bucket>/<object-key>")
        return type('ParsedOSSPath', (object,), {
            'bucket': parsed.netloc,
            'object_key': parsed.path.lstrip('/')
        })

    @oss_retry()
    def _oss_read(self, bucket_name, object_key):
        """OSS 文件读取核心逻辑"""
        auth = oss2.Auth(self.oss_access_key_id, self.oss_access_key_secret)
        bucket = oss2.Bucket(auth, self.oss_endpoint, bucket_name)
        try:
            result = bucket.get_object(object_key)
            return result.read()
        except oss2.exceptions.NoSuchKey:
            # raise FileNotFoundError(f"OSS 文件 {object_key} 不存在")
            return ''
        except oss2.exceptions.OssError as e:
            raise RuntimeError(f"OSS 读取失败: {str(e)}")

    @oss_retry()
    def _oss_write(self, bucket_name, object_key, content):
        """OSS 文件写入核心逻辑（覆盖模式）"""
        auth = oss2.Auth(self.oss_access_key_id, self.oss_access_key_secret)
        bucket = oss2.Bucket(auth, self.oss_endpoint, bucket_name)
        try:
            bucket.put_object(object_key, content)
            return True
        except oss2.exceptions.AccessDenied:
            raise PermissionError(f"无权限写入 OSS 对象 {object_key}")
        except oss2.exceptions.OssError as e:
            raise RuntimeError(f"OSS 写入失败: {str(e)}")

    def delete_file(self, file_path):
        """删除文件（支持本地和OSS）"""
        if file_path.startswith("oss://"):
            return self._delete_oss_file(file_path)
        else:
            return self._delete_local_file(file_path)
    
    def _delete_local_file(self, file_path):
        """删除本地文件"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"本地文件 {file_path} 不存在")
            
            if os.path.isdir(file_path):
                raise IsADirectoryError(f"{file_path} 是目录，请使用删除目录方法")
                
            os.remove(file_path)
            return True
        except PermissionError:
            raise PermissionError(f"无权限删除本地文件 {file_path}")
        except Exception as e:
            raise RuntimeError(f"删除本地文件失败: {str(e)}")

    @oss_retry()
    def _delete_oss_file(self, oss_path):
        """删除OSS文件"""
        parsed_url = self._parse_oss_path(oss_path)
        auth = oss2.Auth(self.oss_access_key_id, self.oss_access_key_secret)
        bucket = oss2.Bucket(auth, self.oss_endpoint, parsed_url.bucket)
        
        try:
            # 检查是否是文件夹标记（以/结尾）
            if parsed_url.object_key.endswith('/'):
                # 删除文件夹需要先删除其中所有对象
                return self._delete_oss_folder(bucket, parsed_url.object_key)
            else:
                bucket.delete_object(parsed_url.object_key)
                return True
        except oss2.exceptions.NoSuchKey:
            raise FileNotFoundError(f"OSS 文件 {parsed_url.object_key} 不存在")
        except oss2.exceptions.AccessDenied:
            raise PermissionError(f"无权限删除 OSS 文件 {parsed_url.object_key}")
        except oss2.exceptions.OssError as e:
            raise RuntimeError(f"OSS 文件删除失败: {str(e)}")

    @oss_retry()
    def _delete_oss_folder(self, bucket, folder_prefix):
        """删除OSS虚拟文件夹及其内容"""
        try:
            # 列出文件夹下所有对象
            objects_to_delete = []
            for obj in oss2.ObjectIterator(bucket, prefix=folder_prefix):
                objects_to_delete.append(obj.key)
            
            if not objects_to_delete:
                raise FileNotFoundError(f"OSS 文件夹 {folder_prefix} 不存在或为空")
            
            # 批量删除
            bucket.batch_delete_objects(objects_to_delete)
            return True
        except oss2.exceptions.OssError as e:
            raise RuntimeError(f"OSS 文件夹删除失败: {str(e)}")

    def exists(self, file_path):
        """检查文件/对象是否存在（支持本地和OSS）"""
        if file_path.startswith("oss://"):
            return self._oss_exists(file_path)
        else:
            return self._local_exists(file_path)
    
    def _local_exists(self, file_path):
        """检查本地文件是否存在"""
        try:
            return os.path.exists(file_path) and not os.path.isdir(file_path)
        except Exception as e:
            raise RuntimeError(f"检查本地文件存在性失败: {str(e)}")
            
    @oss_retry()
    def _oss_exists(self, oss_path):
        """检查OSS文件/对象是否存在"""
        parsed_url = self._parse_oss_path(oss_path)
        auth = oss2.Auth(self.oss_access_key_id, self.oss_access_key_secret)
        bucket = oss2.Bucket(auth, self.oss_endpoint, parsed_url.bucket)
        
        try:
            # 如果是检查文件夹（以/结尾）
            if parsed_url.object_key.endswith('/'):
                return self._oss_folder_exists(bucket, parsed_url.object_key)
            return bucket.object_exists(parsed_url.object_key)
        except oss2.exceptions.OssError as e:
            raise RuntimeError(f"检查OSS对象存在性失败: {str(e)}")

    @oss_retry()
    def _oss_folder_exists(self, bucket, folder_prefix):
        """检查OSS虚拟文件夹是否存在"""
        try:
            # 列出文件夹下是否有对象
            for obj in oss2.ObjectIterator(bucket, prefix=folder_prefix, delimiter='/', max_keys=1):
                return True
            return False
        except oss2.exceptions.OssError:
            return False