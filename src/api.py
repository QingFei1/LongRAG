      
import json
import re
from zhipuai import ZhipuAI
from openai import OpenAI
import backoff
import time
from openai import OpenAI
import httpx
import yaml
with open("../config/config.yaml", "r") as file:
    config = yaml.safe_load(file)["api"]

from zhipuai import ZhipuAI
client = ZhipuAI(api_key=config["zp_key"]) 
def glm(prompt,model,max_tokens):
    try:
        prompt=[{"role": "user", "content":prompt}]
        response = client.chat.completions.create(
        model=model,
        temperature = 0.99,
        messages=prompt,
        max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
            print(f"An error occurred: {e}")  
            time.sleep(0.5)
            return None


client_gpt = OpenAI(
    base_url=config["openai_base_url"], 
    api_key=config["openai_key"],
    http_client=httpx.Client(
        base_url=config["openai_base_url"],
        follow_redirects=True,
    ),
)


def gpt(prompt,model,max_tokens): 
    try:
        message = [{"role": "user", "content": prompt}]
        completion = client_gpt.chat.completions.create(model= model,messages= message,max_tokens=max_tokens, temperature=1)      
        response = completion.choices[0].message.content
        return response

    except Exception as e:
        print(f"An error occurred: {e}")  
        time.sleep(1)
    return None

# Remove duplicate sentences to prevent GPT API from refusing to respond due to repeated input

def remove_consecutive_repeated_sentences(text, threshold=5):
    sentences = re.split(r'([。！？,，])', text)
    
    cleaned_sentences = []
    current_sentence = None
    count = 0

    for i in range(0, len(sentences), 2):
        sentence = sentences[i].strip()
        if i + 1 < len(sentences):
            delimiter = sentences[i + 1]
        else:
            delimiter = ''

        if sentence == current_sentence:
            count += 1
        else:
            if count >= threshold:
                cleaned_sentences.append(current_sentence + delimiter)
            elif current_sentence:
                cleaned_sentences.extend([current_sentence + delimiter] * count)
            current_sentence = sentence
            count = 1

    if count >= threshold:
        cleaned_sentences.append(current_sentence + delimiter)
    else:
        cleaned_sentences.extend([current_sentence + delimiter] * count)
    
    cleaned_text = ''.join(cleaned_sentences)
    return cleaned_text


@backoff.on_exception(backoff.expo, (Exception), max_time=500)
def call_api(prompt,model,max_new_tokens):
    if "glm" in model:
        res=glm(prompt,model, max_new_tokens)
    elif "gpt" in model:
        res=gpt(prompt,model,max_new_tokens)
        if not res:
            prompt=remove_consecutive_repeated_sentences(prompt)
            res=gpt(prompt,model,max_new_tokens)
    assert res != None
    return res

if __name__ == "__main__":

    print(call_api("Hello","gpt-3.5-turbo",100))
    print(call_api("Hello","gpt-3.5-turbo-16k",100))
    print(call_api("Hello","glm-4",100))
    print(call_api("Hello","chatglm_turbo",100))
    



    