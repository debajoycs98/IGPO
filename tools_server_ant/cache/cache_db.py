import lmdb
import dill
from typing import Any, Optional

class LMDBCache:
    def __init__(self, path: str = '/data3/yuying/cache', map_size: int = 1099511627776,  # 1T
                 max_dbs: int = 1, readonly: bool = False):
        """
        :param path: 数据库文件路径
        :param map_size: 最大数据库大小（字节）
        :param readonly: 是否只读模式
        """
        self.readonly = readonly
        self.env = lmdb.open(
            path,
            map_size=map_size,
            max_dbs=max_dbs,
            readonly=readonly,
            sync=True
        )
        stats = self.env.stat()
        entry_count = stats["entries"]
    
        print(f"{path}数据库中共有 {entry_count} 条数据")
        
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
    
    def set(self, sid, value):
        if not self.readonly:
            txn = self.env.begin(write=True)
            txn.put(str(sid).encode(), dill.dumps(value))
            txn.commit()

    def delete(self, sid):
        if not self.readonly:
            txn = self.env.begin(write=True)
            txn.delete(str(sid).encode())
            txn.commit()
    
    def get(self, sid):
        txn = self.env.begin()
        value = txn.get(str(sid).encode())
        if value:
            value = dill.loads(value)
        return value

    def close(self) -> None:
        """关闭环境并释放资源"""
        self.env.close()

