import pickle
import traceback
import re
import multiprocess
from multiprocess import Process, Queue
from typing import Any, Dict
from tqdm import tqdm


# 定义执行代码的worker函数（用于子进程）
def _execute_code_worker(code: str, queue: Queue):
    try:
        runtime = GenericRuntime()
        runtime.inject({"_output": None})
        runtime.exec_code(code)
        result = runtime.output
        if result is None:
            result = "no _output error: The absence of the '_output' variable in the code may have resulted in no valid output."

        # 序列化检查
        str(result)
        pickle.dumps(result)

        queue.put((result, "Done"))
    except:
        err_msg = "code execution error:\n" + str(traceback.format_exc())
        queue.put(("", err_msg))


class GenericRuntime:
    def __init__(self, global_dict={}, headers=[]):
        self._global_vars = global_dict
        for c in headers:
            self.exec_code(c)

    def exec_code(self, code_piece: str) -> None:
        exec(code_piece, self._global_vars)

    def eval_code(self, expr: str) -> Any:
        return eval(expr, self._global_vars)

    def inject(self, var_dict: Dict[str, Any]) -> None:
        for k, v in var_dict.items():
            self._global_vars[k] = v

    @property
    def output(self):
        return self._global_vars.get("_output")


class LocalPythonExecutor:
    def __init__(
        self,
        timeout_length: int = 30,  # 超时时间（秒）
    ) -> None:
        self.timeout_length = timeout_length
        self._tools = {}

    def execute(self, code, tools):
        # 检查代码是否使用tools中的变量
        uses_tools = False
        for tool_name in tools.keys():
            if re.search(r"\b" + re.escape(tool_name) + r"\b", code):
                uses_tools = True
                break

        if uses_tools:
            # 使用tools时直接执行（无超时）
            return self._execute_safely(code, tools)
        else:
            # 未使用tools时使用子进程执行（带超时）
            return self._execute_with_timeout(code)

    def _execute_safely(self, code, tools):
        """安全执行（无超时）"""
        try:
            runtime = GenericRuntime()
            if len(tools) > 0:
                runtime.inject(tools)

            runtime.inject({"_output": None})
            runtime.exec_code(code)
            result = runtime.output
            if result is None:
                result = "no _output error: The absence of the '_output' variable in the code may have resulted in no valid output."

            report = "Done"
            str(result)
            pickle.dumps(result)
        except:
            result = ""
            report = "code execution error:\n" + str(traceback.format_exc())
            print(report)
        return result, report

    def _execute_with_timeout(self, code):
        """使用子进程执行（带超时）"""
        queue = Queue()
        p = Process(target=_execute_code_worker, args=(code, queue))
        p.start()

        # 等待进程完成或超时
        p.join(self.timeout_length)

        # 检查进程是否超时
        if p.is_alive():
            p.terminate()
            p.join()
            return "", "Timeout Error"

        # 获取子进程结果
        if not queue.empty():
            return queue.get()
        return "", "Child process did not return result"

    def __call__(self, code):
        return self.batch_apply([code])[0]

    @staticmethod
    def truncate(s, max_length=10000):
        half = max_length // 2
        if len(s) > max_length:
            s = s[:half] + "..." + s[-half:]
        return s

    def batch_apply(self, all_code_snippets):
        all_exec_results = []

        if len(all_code_snippets) > 100:
            progress_bar = tqdm(total=len(all_code_snippets), desc="Execute")
        else:
            progress_bar = None

        for code in all_code_snippets:
            try:
                result = self.execute(code, self._tools)
                all_exec_results.append(result)
            except Exception as e:
                print("code execution error:\n" + str(traceback.format_exc()))
                all_exec_results.append(
                    ("", "code execution error:\n" + str(traceback.format_exc()))
                )

            if progress_bar is not None:
                progress_bar.update(1)

        if progress_bar is not None:
            progress_bar.close()

        batch_results = []
        for code, (res, report) in zip(all_code_snippets, all_exec_results):
            res, report = str(res).strip(), str(report).strip()
            res, report = self.truncate(res), self.truncate(report)
            if report != "Done":
                batch_results.append(report)
            else:
                batch_results.append(res)
        return batch_results

    def send_tool(self, tools: Dict[str, Any]) -> None:
        self._tools.update(tools)
