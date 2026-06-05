
import glob
import re
import json
from datetime import datetime

LOG_FILES = [
    "/home/admincsp/conversational_rag/logs/rag_server.log",
    "/home/admincsp/conversational_rag/logs/rag_server_8060.log",
    "/home/admincsp/conversational_rag/logs/rag_server_gemini.log",
    "/home/admincsp/conversational_rag/logs/rag_server_ls.log",
    "/home/admincsp/conversational_rag/logs/rag_server_8088.log",
    "/home/admincsp/conversational_rag/logs/rag_server_8088_success.log"
]

START_DATE = datetime(2026, 1, 17)

def get_queries():
    unique_queries = set()
    queries_list = []

    for log_file in LOG_FILES:
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    # Match timestamp and QUERY_START
                    match = re.search(r'^(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}).*QUERY_START \| ({.*})', line)
                    if match:
                        date_str, time_str, json_str = match.groups()
                        log_date = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
                        
                        if log_date >= START_DATE:
                            try:
                                data = json.loads(json_str)
                                query = data.get("query")
                                if query and query not in unique_queries:
                                    unique_queries.add(query)
                                    queries_list.append(query)
                            except:
                                pass
        except FileNotFoundError:
            pass

    return queries_list

if __name__ == "__main__":
    queries = get_queries()
    print(f"Found {len(queries)} unique queries since {START_DATE.date()}")
    print("-" * 50)
    for i, q in enumerate(queries[:20], 1):
        print(f"{i}. {q}")
