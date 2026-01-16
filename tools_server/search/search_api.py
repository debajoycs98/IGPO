import requests
import json
import http.client
import time
import re
from typing import Optional,List,Dict
from unidecode import unidecode


def clean_query(text):
    # 用正则表达式将文本分成中文部分和非中文部分
    # 中文部分保留，非中文部分做 unidecode 处理
    processed = []
    for segment in re.split(r'([\u4e00-\u9fa5]+)', text):  # 按是否是中文分段
        if re.match(r'[\u4e00-\u9fa5]', segment):  # 如果是中文
            processed.append(segment)
        else:  # 非中文部分处理
            cleaned_segment = unidecode(segment)  # 去除重音
            cleaned_segment = cleaned_segment.replace("'", "").replace('"', "")  # 去掉引号
            processed.append(cleaned_segment)
    return ''.join(processed)

def local_wiki_search(query: str, config: Dict):
    def _passages2string(retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference
    search_url = config["wiki_search_url"]
    topk = config["wiki_search_topk"]
    payload = {
        "queries": [query],
        "topk": topk,
        "return_scores": True
    }
    resp = requests.post(search_url, json=payload).json()
    result = resp['result'][0]
    return _passages2string(result)

        
def web_search(query, config):
    if not query:
        raise ValueError("Search query cannot be empty")
    if config['search_engine'] == 'google':
        return serper_data_serp_api(
            query=query,
            region=config['search_region'],
            lang=config['search_lang']
        )
        # return ace_data_serp_api(
        #     query=query,
        #     region=config['search_region'],
        #     lang=config['search_lang']
        # )
    elif config['search_engine'] == 'bing':
        return azure_bing_search(
            query=query,
            subscription_key=config['azure_bing_search_subscription_key'],
            mkt=config['azure_bing_search_mkt'],
            top_k=config['search_top_k']
        )
    elif config['search_engine'] == 'grt':
        return grt_search(
            query=query,
            top_k=config['search_top_k'],
            region=config['search_region'],
            lang=config['search_lang']
        )
    elif config['search_engine'] == 'ant':
        return ant_search(
            query=query,
            top_k=config['search_top_k'],
        )


def azure_bing_search(query, subscription_key, mkt, top_k, depth=0):
    params = {'q': query, 'mkt': mkt, 'count': top_k}
    headers = {'Ocp-Apim-Subscription-Key': subscription_key}

    results = []

    try:
        response = requests.get("https://api.bing.microsoft.com/v7.0/search", headers=headers, params=params)
        json_response = response.json()
        for e in json_response['webPages']['value']:
            results.append({
                "title": e['name'],
                "link": e['url'],
                "snippet": e['snippet']
            })
    except Exception as e:
        print(f"Bing search API error: {e}")
        if depth < 1024:
            time.sleep(1)
            return azure_bing_search(query, subscription_key, mkt, top_k, depth+1)
    return results


def serper_data_serp_api(query,
                      region='',
                      lang=''):
    url = "https://api.acedata.cloud/serp/google"
    headers = { 'X-API-KEY': '1a8150c277e7644e5b8a68822c2e5a4cbd4a9eb3', 'Content-Type': 'application/json' } 
    payload = json.dumps({
      "q": query,
      # "location": "",
      "gl": region,
      "hl": lang,
      "num": 10,
      # "tbs": "qdr:d"
    })
    attempts = 3
    for attempt in range(attempts):
        try:
            conn = http.client.HTTPSConnection("google.serper.dev", timeout=20) 
            conn.request("POST", "/search", payload, headers) 
            res = conn.getresponse() 
            data = json.loads(res.read().decode("utf-8"))
            if 'organic' not in data.keys():
                raise Exception(f"No results found for query: '{query}' '{str(data)}'. Use a less specific query.")
            else:
                results = data["organic"]
                return results
        except Exception as e:
            print(f"serper search API error: {e}")
            if attempt < attempts - 1:
                continue
            else:
                return []

def ace_data_serp_api(query,
                      region='US',
                      lang='en'):
    url = "https://api.acedata.cloud/serp/google"
    headers = {
        'accept': 'application/json',
        'authorization': 'Bearer {}'.format('bdee88e992d34185aca291674cd04e63'),
        'content-type': 'application/json'
    }
    data = {
        "action": "search",
        "query": query,
        "country": region,
        "language": lang
    }
    proxies = {
        'https': 'socks5://127.0.0.1:11080', # SOCKS代理
    }
    attempts = 3
    for attempt in range(attempts):
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data), proxies=proxies, timeout = 30)
            data = response.json()
            if 'organic' not in data.keys():
                raise Exception(f"No results found for query: '{query}' '{str(data)}'. Use a less specific query.")
            else:
                results = data["organic"]
                return results
        except Exception as e:
            print(f"ACE search API error: {e}")
            if attempt < attempts - 1:
                continue
            else:
                return []

def grt_search(
        query, 
        top_k,
        region,
        lang,
        depth=0
    ):
    """
    Perform a web search using the bing google quark etc.
    """
    url = 'https://function.alipay.com/webapi/function/exe'
    headers = {'content-type': 'application/json'}
    env = "PROD"
    
    query = clean_query(query)  # 去掉单双引号以及重音符号导致下游出现语法兼容问题

    data = {
        "functionName": "myjf.common.antseccopilot.seccopilot.agent.internetretrieval",
        "env": env,
        "params": {
            "userId":"200324",
            "source":"factcheck",
            "query":query,
            "count":str(top_k)
        }
    }

    search_response = requests.post(url, headers=headers, data=json.dumps(data))
    
    if search_response.status_code != 200:
        raise Exception(f"Error occurred while searching: {search_response.text}")
    try:
        search_results = json.loads(search_response.json()['data'])
        if len(search_results) == 0:
            time.sleep(1)
            search_response = requests.post(url, headers=headers, data=json.dumps(data))
            search_results = json.loads(search_response.json()['data'])
        if len(search_results) == 0:
            return []
        else:
            formatted_results = ""
            results = []
            i = 0
            # print(query)
            # print(search_results[:3])
            formatted_results = [x for x in search_results]
            # formatted_results = [x for x in search_results if x['searchStrategy'] in ['bing','google','quark']]
            for e in formatted_results:
                e['title'] = e['sourceTitle']
                del e['sourceTitle']
                e['snippet'] = e['abstractText']
                del e['abstractText']
                e['link'] = e['sourceUrl']
                del e['sourceUrl']
        return formatted_results
    except Exception as e:
        print(f"grt search API error: {e}")
        print("search failed")
        return []

def ant_search(
        query, 
        top_k = 10,
    ):
    # https://yuque.antfin.com/amycam/vd89t8/ipihgci68cvdp4lg#vPn9w
    url = 'https://antragflowInside.alipay.com/v1/rpc/ragLlmSearch'
    headers = {
        'Content-Type': 'application/json'
    }

    query = clean_query(query)  # 去掉单双引号以及重音符号导致下游出现语法兼容问题
    
    data = {
        # "domain": "all",
        "domain": "google",
        "extParams": {"needMsResult":"true",
                     "use_safe_whitelist_switch":"false",
                     "includeSites":''},
        "page": 1,
        "pageSize": top_k,
        "query": query,
        "searchMode": "RAG_LLM",
        "source": "factcheck",
        "userId": "2088502287679891"
    }

    result_list = []
    try:
        # response = requests.post(url, headers=headers, data=json.dumps(data))
        # time.sleep(0.5)
        # if response.status_code == 200:
        #     result = response.json()['feedInfo']['feeds']
        #     result_list = []
        #     for e in result:
        #         e2 = {}
        #         e2['title'] = e['extInfo']['title']
        #         e2['snippet'] = e['extInfo']['abstract_extract']
        #         e2['link']  = e['extInfo'].get('action', "")
        #         result_list.append(e2)
        #     return result_list
        trytime = 0
        while(len(result_list) == 0 and trytime < 2):
            response = requests.post(url, headers=headers, data=json.dumps(data))
            time.sleep(0.5)
            if response.status_code == 200:
                result = response.json()['feedInfo']['feeds']
                for e in result:
                    e2 = {}
                    e2['title'] = e['extInfo']['title']
                    e2['snippet'] = e['extInfo']['abstract_extract']
                    e2['link']  = e['extInfo'].get('action', "")
                    result_list.append(e2)
            else:
                print(f"请求失败，状态码: {response.status_code}")
                print("响应内容：", response.text)
            trytime += 1
    except requests.exceptions.RequestException as e:
        # 处理请求过程中的异常（如网络错误）
        print("请求过程中发生错误：", e)
        return []
    return result_list
    

def extract_full_hostname(url):
    # 正则匹配主机名
    match = re.search(r'(?:https?://)?([^:/\s]+)', url)
    if match:
        hostname = match.group(1)
        # 去掉端口号
        return hostname.split(':')[0]
    return None

def serper_google_search_old(
        query, 
        serper_api_key,
        top_k,
        region,
        lang,
        depth=0
    ):
    try:
        conn = http.client.HTTPSConnection("google.serper.dev")
        payload = json.dumps({
                "q": query,
                "num": top_k,
                "gl": region,
                "hl": lang,
            })
        headers = {
            'X-API-KEY': serper_api_key,
            'Content-Type': 'application/json'
        }
        conn.request("POST", "/search", payload, headers)
        res = conn.getresponse()
        data = json.loads(res.read().decode("utf-8"))
        print('data',data)
        if not data:
            raise Exception("The google search API is temporarily unavailable, please try again later.")

        if "organic" not in data:
            raise Exception(f"No results found for query: '{query}'. Use a less specific query.")
        else:
            results = data["organic"]
            print("search success")
            return results
    except Exception as e:
        # print(f"Serper search API error: {e}")
        if depth < 512:
            time.sleep(1)
            return serper_google_search(query, serper_api_key, top_k, region, lang, depth=depth+1)
    print("search failed")
    return []


if __name__ == "__main__":
    print(serper_google_search("test", "your serper key",1,"us","en"))