
    
    def post_request(self, server_url: str, query_contents: List[dict],server_cnt):
        depth = 0
        if not query_contents:
            return query_contents
        while True:
            try:
                response = requests.post(f"{server_url}/handle_execution", json={"query_contents": query_contents}, timeout=999)
                return response.json()['query_contents']
            except Exception as e:
                print(f"{server_cnt} Error occurred: {e}",flush=True)
                depth += 1
                time.sleep(1)
 