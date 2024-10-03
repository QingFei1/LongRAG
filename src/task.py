
import random
from transformers import AutoTokenizer
import re
import json
import yaml
with open("../config/config.yaml", "r") as file:
    config = yaml.safe_load(file)
# Use chatglm3-6b-32k to calculate the number of tokens
model2path = config["model_path"]["chatglm3-6b-32k"]
tokenizer = AutoTokenizer.from_pretrained(model2path, trust_remote_code=True)

from api import call_api
    
def get_word_len(input):
    tokenized_prompt = tokenizer(input, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
    return len(tokenized_prompt)


def build_ext_instruction(model, question, answer, content, support, min_res_tokens):
    support="\n".join(support)
    prompt = f"""{support}.\nBased on the above background only, please output the original information that needs to be cited to answer the following questions. Please ensure that the information cited is detailed and comprehensive.\nQuestion: {question}.\nOutput only the original information of the required reference:"""
    try:
        response = call_api(prompt, model, 1000)

        if get_word_len(response) < min_res_tokens:
            return None
    except:
        return None

    verify_prompt = f"""I am going to provide you a question, the background information, and the answer to that question. Please evaluate whether the answer can be solely derived from the given background information. If it can, set the status value as True, if it can't, set the status value as False.\nQuestion: {question}\nBackground Information: {response}\nAnswer: {answer}\nYour output format should be the following json format: {{"status": "{{the value of status}}"}}"""
    succeed=False
    try:
        flag = call_api(verify_prompt, model, 1000)
        if json.loads(flag)["status"].lower() == "true":
            succeed=True
    except:
        if re.search("True|true",flag):
            succeed=True
    if succeed:
        return {
            "instruction": f"{content}\n\nBased on the above background, please output the information you need to cite to answer the question below.\n{question}",
            "input": "",
            "output": response
        }

    return None

def build_cot_instruction(model, question, answer, content, support, min_res_tokens):
    support="\n".join(support)
    prompt = f"""{support}\n\nGiven question: {question}\nThe answer is: {answer}\nYour task is to give your thought process for this given question based on the above information, only give me your thought process and do not output other information.\nThought process:"""

    try:
        response = call_api(prompt, model, 1000)
        if get_word_len(response) < min_res_tokens:
            return None
    except:
        return None

    verify_prompt = f"""Question: {question},\nThought process of the question: {response}\nAnswer: {answer}\nPlease evaluate whether the thought process of this question can explain the answer to this question. If it can explain the answer, set the value of status to "True". If it cannot explain the answer, set the value of status to "False".\nYour output format should be the following json format: {{"status": "{{the value of status}}"}}"""
    succeed=False
    try:
        flag = call_api(verify_prompt, model, 1000)
        if json.loads(flag)["status"].lower() == "true":
            succeed=True
    except:
        if re.search("True|true",flag):
            succeed=True
    if succeed:
        return {
            "instruction": f"{content}\nPlease combine the above information and give your thought process for the following question:\n{question}.",
            "input": "",
            "output": response
        }

    return None

def build_rag_instruction(model, question, answer, content):
    return {
        "instruction": f"{content}\nBased on the above information, only give me the answer and do not output any other words.\nQuestion: {question}\nAnswer: ",
        "input": "",
        "output": answer
    }

def build_fil_instruction(model, question, answer, support, non_support, sup_flag):
    support_paragraphs=support
    non_paragraphs=non_support
    support="\n".join(support)
    prompt = f"""{support}\n\nGiven question: {question}\nThe answer is: {answer}\nYour task is to give your thought process for this given question based on the above information, only give me your thought process and do not output other information.\nThought process:"""

    try:
        response = call_api(prompt, model, 1000)
    except:
        return None

    verify_prompt = f"""Question: {question},\nThought process of the question: {response}\nAnswer: {answer}\nPlease evaluate whether the thought process of this question can explain the answer to this question. If it can explain the answer, set the value of status to "True". If it cannot explain the answer, set the value of status to "False".\nYour output format should be the following json format: {{"status": "{{the value of status}}"}}"""
    try:
        flag = call_api(verify_prompt, model, 1000)
        if json.loads(flag)["status"].lower() != "true":
            return None
    except:
        if not re.search("True|true",flag):
            return None

    if sup_flag :
        content = random.sample(support_paragraphs, 1)[0]
        status = '{"status":"True"}'
    else:
        content = random.sample(non_paragraphs, 1)[0]
        status = '{"status":"False"}'

    return {
        "instruction": f"""Given an article: {content}\nQuestion: {question}.\nThought process for the question: {response}\nYour task is to use the thought process provided to decide whether you need to cite the article to answer this question. If you need to cite the article, set the status value to True. If not, set the status value to False. Please output the response in the following json format: {{"status": "{{the value of status}}"}}""",
        "input": "",
        "output": status
    }