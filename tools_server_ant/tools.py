import time
import json
import concurrent.futures
import threading
import yaml
from typing import List, Dict
from smolagents import Tool
from openai import OpenAI
from loguru import logger
import traceback
import hashlib
import uuid
import requests
from types import SimpleNamespace
import random

from tools_server.util import FileSystemReader
from tools_server.cache.cache_db import LMDBCache


class LocalWikiSearchTool(Tool):
    name = "local_wiki_search"
    description = "This is a wiki retriever, which can retrieve relevant information snippets from wiki through keywords."
    inputs = {
        "query": {
            "type": "string",
            "description": "Can query specific keywords or topics to retrieve accurate and comprehensive information. The query is preferably English keywords.",
        }
    }
    example = {"name": "local_wiki_search", "arguments": {"query": "xxxx"}}
    output_type = "string"

    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

    def forward(self, query:str) -> str:
        from tools_server.search.search_api import local_wiki_search
        try:
          result = local_wiki_search(query, self.config)
          return result
        except Exception as e:
            raise Exception(f"Error occurred while parsing wiki search results: {str(e)}")


class WebSearchTool(Tool):
    name = "web_search"
    description = "Search the web for relevant information from google. the tool retrieves the top 10 results for each query in one call."
    inputs = {
        "query": {
            "type": "array",
            "description": "The queries to search, which helps answer the question",
        }
    }
    
    example = {"name": "web_search", "arguments": {"query": ["xxxx","yyyy"]}}
    output_type = "array"

    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.search_cache_path = self.config['search_cache_path']
        self.search_cache = LMDBCache(self.search_cache_path, readonly=config['cache_readonly'])

    def forward(self, query) -> Dict:
        if type(query) == str:
            query = [query]
        from tools_server.search.search_api import web_search
        results = []
        for q in query:
            query_cache = self.search_cache.get(q)
            
            if query_cache:
                print('命中cache:', q)
                if not self.config['cache_readonly'] and random.random() < 0.1:
                    query_cache['last_hit_timestamp'] = time.time()
                    self.search_cache.set(q, query_cache)
                
            if query_cache and query_cache['organic'] and len(query_cache['organic']) > 0:
                organic = query_cache['organic']
            else:
                organic = web_search(q, self.config)
                self.search_cache.set(q, {
                    'timestamp': time.time(),  # creation timestamp
                    'last_hit_timestamp': time.time(),  # first access is also a hit
                    'organic': organic
                })
    
            ret_web_page_info_list = []
            for web_info in organic:
                ret_web_page_info_list.append({
                    "title": web_info['title'],
                    "url": web_info['link'],
                    "quick_summary": web_info['snippet'] if 'snippet' in web_info else "",
                    "date": web_info['date'] if 'date' in web_info else "",
                })
            results.append(str({"search_query": q, "web_page_info_list": ret_web_page_info_list}))
        return "\n=======\n".join(results)


class BrowseWebpageTool(Tool):
    name = "browse_webpage"
    description = "Browse the webpage and return the content that not appeared in the conversation history. You should use this tool if the last action is search and the search result maybe relevant to the question."
    inputs = {
        "url_list": {
            "type": "array",
            "description": "The chosen urls from the search result, do not use url that not appeared in the search result. Do not use more than 3 urls.",
        },
        "query": {
            "type": "string",
            "description": "These queries aim to retrieve information from the URL webpage.",
        }
    }
    example = {"name": "browse_webpage", "arguments": {"url_list": ["http://www.dada.com", "xxxx"], "query": "xxxx"}}
    output_type = "array"

    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from tools_server.webbrower.read_agent import ReadingAgent
        self.config = config
        self.client = OpenAI(
            api_key=self.config['openai_api_key'], 
            base_url=self.config["openai_base_url"],
        ) # 已经使用内部接口，可以不用了

        self.read_agent = ReadingAgent(self.config, client = self.client)
        self.fs_reader = FileSystemReader(
            oss_access_key_id = self.config['oss_access_key_id'],
            oss_access_key_secret = self.config['oss_access_key_secret'],
            oss_endpoint = self.config['oss_endpoint']
        )
        self.brower_cache_path = self.config['brower_cache_path']
        if self.brower_cache_path[-1] == '/':
            self.brower_cache_path = self.brower_cache_path[:-1]
        self.brower_cache = LMDBCache(self.brower_cache_path + '/lmdb_file', readonly = False, map_size = 50 * 1024 * 1024 * 1024)
        self.browse_cache_lock = threading.Lock()

    def forward(self, url_list: List[str], query: str):
        from tools_server.webbrower.read_agent import grt_browser, WebPageInfo, EXTRACT_NEW_INFO_PROMPT, get_response_from_llm, get_content_from_tag
        url_dict = {}
        user_query = query
        if type(user_query) == list:
            user_query = user_query[0]
        results = []
        for url in url_list:
            key = user_query + "|" + url
            query_url_result = {}
            url_result = {}
            query_cache = self.brower_cache.get(key)
            if query_cache:
                print('命中cache:',key,query_cache)
                if random.random() < 0.1:
                    query_cache['last_hit_timestamp'] = time.time()
                    self.brower_cache.set(key, query_cache)
            if query_cache and query_cache.get('extracted_info','') != '':
                query_url_result = query_cache
            else:
                url_cache = self.brower_cache.get(url)
                if url_cache:
                    print('命中cache:', url, url_cache)
                if url_cache and url_cache.get('urlText','') != '':
                    url_result['urlText'] = url_cache['urlText']
                    url_result['urlScreenShot'] = url_cache['urlScreenShot']
                else:
                    url_result = grt_browser(url)
                    url_result['timestamp'] = time.time()
                    
                    def url_to_uuid(url: str) -> str:
                        # 使用 SHA-256 哈希 URL
                        hash_obj = hashlib.sha256(url.encode())
                        hex_digest = hash_obj.hexdigest()  # 64 字符的十六进制字符串
                        
                        # 截取前 32 字符（128 位）并格式化为 UUID
                        uuid_str = f"{hex_digest[:8]}-{hex_digest[8:12]}-{hex_digest[12:16]}-{hex_digest[16:20]}-{hex_digest[20:32]}"
                        return uuid_str

                    if url_result.get('screenshot','') != '':
                        try:
                            screenshot_path = self.brower_cache_path + '/' + url_to_uuid(url) + '/screenshot.jpg'
                            response = requests.get(url_result.get('screenshot',''), timeout=10)
                            content = response.content
                            self.fs_reader.write_file(screenshot_path, content)
                            domUrl_path = self.brower_cache_path + '/' + url_to_uuid(url) + '/text.txt'
                            response = requests.get(url_result.get('domUrl',''), timeout=10)
                            content = response.text
                            self.fs_reader.write_file(domUrl_path, content)
                            url_result['urlScreenShot'] = screenshot_path
                            url_result['urlText'] = domUrl_path
                        except requests.exceptions.RequestException as e:
                            print(f"下载失败: {str(traceback.format_exc())}")
                            url_result['urlScreenShot'] = ''
                            url_result['urlText'] = ''
                        if url_result.get('urlScreenShot','') != '':
                            self.brower_cache.set(url, url_result)
                query_url_result['timestamp'] = time.time()
                try:
                    urlText = self.fs_reader.read_file(url_result['urlText']).decode("utf-8") 
                except Exception as e:
                    self.brower_cache.delete(url)
                    print(f"{url}缓存文件没有找到，删除缓存。url_result:{url_result};{str(traceback.format_exc())}")
                    continue
                cur_web_page_content = urlText
                prompt = EXTRACT_NEW_INFO_PROMPT.format(
                    main_question=user_query,
                    context_so_far=user_query,
                    page_index=1,
                    total_pages=1,
                    page_content=cur_web_page_content
                )
                messages = [{"role": "user", "content": prompt}]
                print('调用qwen2.5处理：', messages)
                response = get_response_from_llm(
                    messages=messages,
                    client=self.client,
                    model=self.config["reading_agent_model"],
                    stream=False
                )
                print('qwen2.5处理结果：',response)
                page_thinking = response["reasoning_content"] if "reasoning_content" in response else ""
                query_url_result['page_thinking'] = response["reasoning_content"] if "reasoning_content" in response else ""
                query_url_result['extracted_info'] = get_content_from_tag(response["content"], "extracted_info", "").strip()
                query_url_result['urlText'] = urlText
                query_url_result['timestamp'] = time.time()
                if query_url_result.get('extracted_info','') != '':
                    self.brower_cache.set(key, query_url_result)
            results.append({
                "url": url,
                "screenshotpath": url_result.get('urlScreenShot',''),
                "information": query_url_result['extracted_info']
            })
        return results


class CodeActTool(Tool):
    name = "code_act"
    description = "Execute python code."
    inputs = {
        "code": {
            "type": "string",
            "description": "The python code, and the result must be placed in the '_output' variable.",
        }
    }
    example = {"name": "code_act", "arguments": {"code": "xxxx"}}
    output_type = "string"

    def __init__(self, *args, **kwargs):
        self.additional_authorized_imports = kwargs.get('additional_authorized_imports',[])
        self.tools_add = kwargs.get('tools')
        super().__init__(*args, **kwargs)
        
    def forward(self, code: str) -> str:
        from tools_server.code.local_python_exec import LocalPythonExecutor
        pythonexe = LocalPythonExecutor()
        if self.tools_add and len(self.tools_add)>0:
            pythonexe.send_tool(self.tools_add)
        return pythonexe(code)


def get_tools(config):
    available_tools = config['available_tools']
    tool_classes = {
        "local_wiki_search": LocalWikiSearchTool,
        "web_search": WebSearchTool,
        "browse_webpage": BrowseWebpageTool
    }
    AVAILABLE_TOOLS = {
        name: tool_classes[name]
        for name in available_tools if name in tool_classes
    }
    return AVAILABLE_TOOLS


# 读取调用
class Handler:
    def __init__(self, agent_config):
        self.logger = logger
        self.agent_config = agent_config
        self.fs_reader = FileSystemReader(
            oss_access_key_id = agent_config['oss_access_key_id'],
            oss_access_key_secret = agent_config['oss_access_key_secret'],
            oss_endpoint = agent_config['oss_endpoint']
        )
        self.available_tools = {name: tool(self.agent_config) for name, tool in get_tools(self.agent_config).items()}
        # self.available_tools.update(
        #     {"code_act": CodeActTool(tools=self.available_tools)}
        # )

    def save_cache_to_json(self, cache_dict, cache_path):
        try:
            path = cache_path
            self.fs_reader.write_file(path, json.dumps(cache_dict, indent=4, ensure_ascii=False))
        except Exception as e:
            self.logger(f"Error saving cache: {e}")
 
    # def handle_all(self, contents):
    #     try:  
    #         api_future_list = []
    #         with concurrent.futures.ThreadPoolExecutor(max_workers=10) as api_executor:
    #             for content in contents:
    #                 api_future = api_executor.submit(self.handle_single, content)
    #                 api_future_list.append(api_future)
    #         for i, future in enumerate(api_future_list):
    #             contents[i]["content"] = future.result()

    #         return contents
    #     except Exception as e:
    #         self.logger.error(f"handle_all错误: {str(traceback.format_exc())}")
    #         return contents

    def handle_all(self, contents):
        try:
            api_future_map = {}  # key: str(tool_call), value: future
            tool_call_to_indices = {}  # key: str(tool_call), value: list of indices
    
            # Step 1: 去重并记录索引位置
            for i, content in enumerate(contents):
                tool_call_str = json.dumps(content["tool_call"], sort_keys=True)
                if tool_call_str not in tool_call_to_indices:
                    tool_call_to_indices[tool_call_str] = []
                tool_call_to_indices[tool_call_str].append(i)
    
            # Step 2: 并发执行唯一的调用
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as api_executor:
                for tool_call_str in tool_call_to_indices:
                    tool_call = json.loads(tool_call_str)
                    future = api_executor.submit(self.handle_single, {"tool_call": tool_call})
                    api_future_map[tool_call_str] = future
    
            # Step 3: 收集结果并映射回原始内容
            for tool_call_str, indices in tool_call_to_indices.items():
                result = api_future_map[tool_call_str].result()
                for i in indices:
                    contents[i]["content"] = result
    
            return contents
    
        except Exception as e:
            self.logger.error(f"handle_all错误: {str(traceback.format_exc())}")
            return contents
            
    def handle_single(self, content):
        try:  
            tool_call = content["tool_call"]
            tool_name = tool_call["name"]
            arguments = tool_call['arguments']           
            assert tool_name in self.available_tools, f"invalid tool name{tool_name}, must be one of {self.available_tools}"
            print("tool_call", tool_call)
            tool = self.available_tools[tool_name]
            result = tool(**arguments)
            return result
        except Exception as e:
            self.logger.error(f"handle_single: {str(traceback.format_exc())}")
            return ['工具处理失败']
        
    
