import requests
import json
messages=[]
messages.append({ "role": "user", "content": "您是谁" })
url = "https://api.chatanywhere.com.cn/v1/chat/completions"
import os
import requests
import time
import json
# import openai
from openai import OpenAI
client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="sk-HE9LX33nb5ULscY1S4md8qaErOP4CqucyevfdfogHCuQxZEB",
    base_url="https://api.chatanywhere.tech/v1"
)



# 非流式响应
messages=[]
messages.append({ "role": "user", "content": '你是谁' })
def gpt_35_api(messages: list):
    """为提供的对话消息创建新的回答

    Args:
        messages (list): 完整的对话消息
    """
    completion = client.chat.completions.create(model="gpt-4-0125-preview", messages=messages)
    print(completion.choices[0].message.content)
gpt_35_api(messages)
   
# payload = json.dumps({
#         "model": "gpt-4-0125-preview",
#         "messages": messages,
#         "temperature": 0.7
#         })
# headers = {
#         'Authorization':'sk-HE9LX33nb5ULscY1S4md8qaErOP4CqucyevfdfogHCuQxZEB',
#         'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
#         'Content-Type': 'application/json'
#         }
    

# stream = requests.request("POST", url, headers=headers, data=payload)
# print(chat_completions())