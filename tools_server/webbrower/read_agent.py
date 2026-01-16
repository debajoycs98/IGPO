from typing import List, Dict, Any, Optional
import time
import random
import html2text
import concurrent.futures
from .text_web_browser import SimpleTextBrowser
from types import SimpleNamespace
import re
from urllib.parse import urlparse
import time
import requests
import json
from multiprocessing import Pool
from functools import partial
from openai import OpenAI
import difflib
import string
import os
import traceback

def internal_llm(query):
    url = 'https://lingyumng-prod.alipay.com/dataset/model_answer'
    # data = {
    #   "model": "qwen-plus", 
    #   "workNo": "242562", 
    #   "question": query, 
    #   "token": "af1ad4ff70fb4a3eb74c272b7bb2ae5febd0758417d84736b611ac1906d045b2bdbe68ecec744537976f1471f5203606" 
    # }
    data = {
      "model": "qwen-turbo", ## 可替换为申请的模型名称
      "workNo": "242562", #替换为自己的工号
      "question": query, #传入提问的问题
      "token": "7361a63984574ee08fed8b2cf5b1a9d6cf08d492ed2842a1bb500035c4965794f7876ab268fd40739ba2d96c3d6ee18b" #填入申请好的权限码
    }
    json_data = json.dumps(data)
    headers = {"Content-Type": "application/json"}
    try:
      response = requests.post(url, data=json_data, headers=headers)
      if response.headers.get('Content-Type') == 'application/json' and response.json()['success']:
          return response.json()['data']['reply']
      else:
          return None
    except Exception as e:
      print(e)
      return None


def extract_url_root_domain(url):
    """
    从 URL 中提取根域名
    例如:
    - https://www.example.com/path -> example.com
    - sub.example.co.uk -> example.co.uk
    """
    # 确保 URL 包含协议，如果没有则添加
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url
    
    # 使用 urlparse 解析 URL
    parsed = urlparse(url).netloc
    if not parsed:
        parsed = url
        
    # 移除端口号(如果存在)
    parsed = parsed.split(':')[0]
    
    # 分割域名部分
    parts = parsed.split('.')
    
    # 处理特殊的二级域名，如 .co.uk, .com.cn 等
    if len(parts) > 2:
        if parts[-2] in ['co', 'com', 'org', 'gov', 'edu', 'net']:
            if parts[-1] in ['uk', 'cn', 'jp', 'br', 'in']:
                return '.'.join(parts[-3:])
    
    # 返回主域名部分（最后两部分）
    return '.'.join(parts[-2:])

def get_clean_content(line):
    clean_line = re.sub(r'^[\*\-•#\d\.]+\s*', '', line).strip()
    clean_line = re.sub(r'^[\'"]|[\'"]$', '', clean_line).strip()
    if (clean_line.startswith('"') and clean_line.endswith('"')) or \
    (clean_line.startswith("'") and clean_line.endswith("'")):
        clean_line = clean_line[1:-1]
    return clean_line

def get_content_from_tag(content, tag, default_value=None):
    # 说明：
    # 1) (.*?) 懒惰匹配，尽量少匹配字符
    # 2) (?=(</tag>|<\w+|$)) 使用前瞻，意味着当后面紧跟 </tag> 或 <任意单词字符开头的标签> 或文本结束时，都停止匹配
    # 3) re.DOTALL 使得点号 . 可以匹配换行符
    pattern = rf"<{tag}>(.*?)(?=(</{tag}>|<\w+|$))"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return default_value


def get_response_from_llm(
        messages,
        client,
        model,
        stream = False,
        temperature = 0.6,
        depth = 0
):
    try:
        content = internal_llm(messages[0]['content'])
        return {
            "content": content.strip()
        }
    except Exception as e:
        print(f"LLM API error: {e}")
        if "Input data may contain inappropriate content" in str(e):
            return {
                "content": ""
            }
        if "Error code: 400" in str(e):
            return {
                "content": ""
            }
        if depth < 3:
            time.sleep(3)
            return get_response_from_llm(messages=messages, client=client, model=model, stream=stream, temperature=temperature, depth=depth+1)
        raise e


EXTRACT_NEW_INFO_PROMPT = """You are a helpful AI research assistant. I will provide you:
* The user's main question. This is a complex question that requires a deep research to answer.
* The context so far. This includes all the information that has been gathered from previous turns, including the sub-questions and the information gathered from other resources for them.
* One page of a webpage content as well as the page index. We do paging because the content of a webpage is usually long and we want to provide you with a manageable amount of information at a time. So please mind the page index to know which page you are reading as this could help you infer what could appear in other pages.

Your task is to read the webpage content carefully and extract all *new* information (compared to the context so far) that could help answer either the main question or the sub-question. So you should only gather incremental information from this webpage, but if you find additional details that can complete the previous context, please include them. If you find contradictory information, also include them for further analysis. Provide detailed information including numbers, dates, facts, examples, and explanations when available. Keep the original information as possible, but you can summarize if needed.

In addition to the extracted information, you should also think about whether we need to read more content from this webpage to get more detailed information by paing down to read more content. Also, add a very short summary of the extracted information to help the user understand the new information.


Note that there could be no useful information on the webpage.

Your answer should follow the following format: 
* Put the extracted new information in <extracted_info> tag. If there is no new information, leave the <extracted_info> tag empty. Do your best to get as much information as possible.
* Put "yes" or "no" in <page_down> tag. This will be used for whether to do page down to read more content from the web. For example, if you find the extracted information is from the introduction section in a paper, then you can infer that the extracted information could miss detailed information, next round can further read more content for details in this web page by paging down. If this already the last page, always put "no" in <page_down> tag.
* Put the short summary of the extracted information in <short_summary> tag. Try your best to make it short but also informative as this will present to the user to notify your progress. If there is no useful new information, please also say something like "Didn't find useful information, will read more" in the short summary (be free to use your own words). 

Important note: Use the same language as the user's main question for the short summary. For example, if the main question is using Chinese, then the short summary should also be in Chinese.


<context_so_far>
{context_so_far}
</context_so_far>

<main_question>
{main_question}
</main_question>

<webpage_content>
    <page_index>{page_index}</page_index>
    <total_page_number>{total_pages}</total_page_number>
    <current_page_content>{page_content}</current_page_content>
</webpage_content>

Now think and extract the incremental information that could help answer the main question or the sub-question."""

def grt_browser(url, timeout=60, interval=2):
    api_key = 'llT6nDGA/QC6ucwxZLfRV3CMtAnt2Qj1'
    send_url = f'https://xhuntercore-prod.alipay.com/xhunterlib/api/v1/g/send?key={api_key}'
    result_url = 'https://xhuntercore-prod.alipay.com/xhunterlib/api/v1/g/result'
    headers = {'Content-Type': 'application/json'}

    # 1. 发起任务
    send_payload = {
        "jobCode": "yuying",
        "properties": {
            "url": url,
            "_jobCode_": "yuying"
        }
    }
    try:
        send_response = requests.post(send_url, headers=headers, json=send_payload)
        send_response.raise_for_status()
        req_id = send_response.json().get("data", {})
        if not req_id:
            print("未获取到reqId")
            return {}
    except Exception as e:
        print(f"发送任务失败: {e}")
        return {}

    # 2. 轮询结果
    max_attempts = timeout // interval
    for _ in range(max_attempts):
        try:
            response = requests.post(result_url, params={"key": api_key, "reqId": req_id})
            response.raise_for_status()
            result_data = response.json()
            if result_data.get("data"):
                return result_data["data"]
        except Exception as e:
            print(f"轮询错误: {e}")
        time.sleep(interval)
    return {}

        
class WebPageInfo:
    def __init__(self,
                 title: str,
                 url: str,
                 quick_summary: str,
                 sub_question,
                 browser: SimpleTextBrowser = None):
        self.title = title
        self.url = url
        self.quick_summary = quick_summary
        self.browser = browser
        self.sub_question = sub_question
        self.page_read_info_list = []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'title': self.title,
            'url': self.url,
            'quick_summary': self.quick_summary,
            # Note: browser object might not be serializable directly, 
            # consider adding a separate serialization method if needed
            'sub_question': self.sub_question,
            'page_read_info_list': [info.to_dict() for info in self.page_read_info_list]
        }
    
    def __str__(self) -> str:
        base_info = f"WebPage: {self.title}\nURL: {self.url}\nQuick Summary: {self.quick_summary}\nSub Question: {self.sub_question}"
        
        if self.page_read_info_list:
            read_info = "\nDetailed Information:"
            for idx, info in enumerate(self.page_read_info_list, 1):
                read_info += f"\n  {idx}. {str(info)}"
            return base_info + read_info
        
        return base_info
        
class ReadingAgent:
    def __init__(self,
                 config,
                 client):
        self.config = config
        self.client = client
        self.user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"

        self.BROWSER_CONFIG = {
            "viewport_size": 1024 * 5 * 8,
            "downloads_folder": "downloads_folder",
            "request_kwargs": {
                "headers": {"User-Agent": self.user_agent},
                "timeout": (5, 10),
            },
            # "serper_api_key": config['serper_api_key'],
        }
        
    def is_error_page(self, browser: SimpleTextBrowser) -> bool:
        if isinstance(browser.page_title, tuple):
            return True
        return (browser.page_title is not None and 
                browser.page_title.startswith("Error ") and 
                browser.page_content is not None and 
                browser.page_content.startswith("## Error "))
        
    def scrape(self, browser, url: str) -> str:
        """爬取网页并使用LLM总结内容"""
        browser.visit_page(url)
        header, content = browser._state()
        return header.strip() + "\n=======================\n" + content
    
    def fetch_content(self, browser: SimpleTextBrowser, url: str):
        try:
            return self.scrape(browser, url)
        except Exception as e:
            return "## Error : No valid information in this page"
            
    def scrape_and_check_valid_api(self, url):
        browser = SimpleTextBrowser(**self.BROWSER_CONFIG)
        content = self.fetch_content(browser, url)
        if content is None:
            return None
        
        if self.is_error_page(browser):
            print(f"访问错误，抛弃URL：{url}")
            return None
        return browser
        
    def read(
            self,
            main_question,
            url,
    ):
        browser = self.scrape_and_check_valid_api(url)
        cur_webpage = WebPageInfo(
                title='',
                url=url,
                quick_summary='',
                browser=browser,
                sub_question=main_question
            )
        if browser is None:
            browser = "error"
            return cur_webpage
        cur_useful_info = ""
        total_pages = len(cur_webpage.browser.viewport_pages)
        last_page = -1
        while cur_webpage.browser.viewport_current_page < total_pages:
            print(f"开始处理页面:{url} ,总共{total_pages}页,当前第{cur_webpage.browser.viewport_current_page}页")
            if cur_webpage.browser.viewport_current_page == last_page:
                break
            context_so_far = ""
            if cur_useful_info:
                context_so_far = f"<main_question>{main_question}</main_question>\n<useful_info>{cur_useful_info}</useful_info>"
            else:
                context_so_far = f"<main_question>{main_question}</main_question>"
            cur_web_page_content = cur_webpage.browser._state()[1]
            cur_web_page_content = html2text.html2text(cur_web_page_content)
            page_index = cur_webpage.browser.viewport_current_page + 1
            prompt = EXTRACT_NEW_INFO_PROMPT.format(
                main_question=main_question,
                context_so_far=context_so_far.strip(),
                page_index=page_index,
                total_pages=total_pages,
                page_content=cur_web_page_content
            )

            messages = [{"role": "user", "content": prompt}]
            # print('调用qwen2.5处理：',messages)
            response = get_response_from_llm(
                messages=messages,
                client=self.client,
                model=self.config["reading_agent_model"],
                stream=False
            )
            # print('qwen2.5处理结果：',response)
            
            extracted_info = get_content_from_tag(response["content"], "extracted_info", "").strip()
            page_down = get_content_from_tag(response["content"], "page_down", "").strip()
            short_summary = get_content_from_tag(response["content"], "short_summary", "").strip()
            print("第",cur_webpage.browser.viewport_current_page,"页：response",response)
            last_page = cur_webpage.browser.viewport_current_page
            if "yes" in page_down:
                page_down = True
            else:
                page_down = False

            if extracted_info:
                cur_webpage.page_read_info_list.append(
                    SimpleNamespace(
                        url=cur_webpage.url,
                        fetch_res=cur_web_page_content,
                        page_thinking=response["reasoning_content"] if "reasoning_content" in response else "",
                        page_summary=extracted_info,
                        page_number=cur_webpage.browser.viewport_current_page,
                        need_page_down=page_down,
                        used=False,
                    )
                )
                cur_useful_info += extracted_info + "\n\n"
                if len(cur_useful_info) > 8000:
                    cur_useful_info = '...' + cur_useful_info

            if page_down:
                print("page_down")
                cur_webpage.browser.page_down()
            else:
                break
        return cur_webpage


