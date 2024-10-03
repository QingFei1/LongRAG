import re
import json
import faiss 
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForSequenceClassification
from transformers.generation.utils import GenerationConfig
import numpy as np
import torch
import os
import random
from sentence_transformers import SentenceTransformer
from datetime import datetime
import backoff
import logging
import argparse
import yaml
from metric import F1_scorer
from api import call_api

logger = logging.getLogger()

choices = [
    "glm-4", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-0125","chatGLM3-6b-32k", "chatGLM3-6b-8k","LongAlign-7B-64k",
    "qwen1.5-7b-chat-32k", "vicuna-v1.5-7b-16k","Llama3-8B-Instruct-8k", "Llama3-70b-8k", "Llama2-13b-chat-longlora",
    "LongRAG-chatglm3-32k", "LongRAG-qwen1.5-32k","LongRAG-vicuna-v1.5-16k", "LongRAG-llama3-8k",  "LongRAG-llama2-4k"
]

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, choices=["hotpotqa", "2wikimultihopqa", "musique"], default="hotpotqa", help="Name of the dataset")
parser.add_argument('--top_k1', type=int, default=100, help="Number of candidates after initial retrieval")
parser.add_argument('--top_k2', type=int, default=7, help="Number of candidates after reranking")
parser.add_argument('--model', type=str, choices=choices, default="chatGLM3-6b-32k", help="Model for generation")
parser.add_argument('--lrag_model', type=str, choices=choices, default="", help="Model for LongRAG")
parser.add_argument('--rb', action="store_true", default=False, help="Vanilla RAG")
parser.add_argument('--raw_pred', action="store_true", default=False, help="LLM direct answer without retrieval")
parser.add_argument('--rl', action="store_true", default=False, help="RAG-Long")
parser.add_argument('--ext', action="store_true", default=False, help="Only using Extractor")
parser.add_argument('--fil', action="store_true", default=False, help="Only using Extractor")
parser.add_argument('--ext_fil', action="store_true", default=False, help="Using Extractor and Filter")
parser.add_argument('--MaxClients', type=int, default=1)
parser.add_argument('--log_path', type=str, default="")
parser.add_argument('--r_path', type=str, default="../data/corpus/processed/200_2_2", help="Path to the vector database")
args = parser.parse_args()



def get_word_len(input):
    tokenized_prompt = set_prompt_tokenizer(input, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
    return len(tokenized_prompt)

def set_prompt(input, maxlen):
    tokenized_prompt = set_prompt_tokenizer(input, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
    if len(tokenized_prompt) > maxlen:
         half = int(maxlen * 0.5)
         input = set_prompt_tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + set_prompt_tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
    return input, len(tokenized_prompt)

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(model2path, model_name):
    if "gpt" in model_name or "glm-4" in model_name or "glm3-turbo-128k" in model_name:
        return model_name, model_name
    if "chatglm" in model_name or "internlm" in model_name or "xgen" in model_name or "longalign-6b" in model_name or "qwen" in model_name or "llama3" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model2path[model_name], trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model2path[model_name], trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
    if "vicuna" in model_name:
        from fastchat.model import load_model
        model, _ = load_model(model2path[model_name], device="cpu", num_gpus=0, load_8bit=False, cpu_offloading=False, debug=False)
        model = model.to(device)
        model = model.bfloat16()
        tokenizer = AutoTokenizer.from_pretrained(model2path[model_name], trust_remote_code=True, use_fast=False)
    elif "llama2" in model_name:
        tokenizer = LlamaTokenizer.from_pretrained(model2path[model_name])
        model = LlamaForCausalLM.from_pretrained(model2path[model_name], torch_dtype=torch.bfloat16, device_map='auto')
    model = model.eval()
    return model, tokenizer

@backoff.on_exception(backoff.expo, (Exception), max_time=200)
def pred(model_name, model, tokenizer, prompt, maxlen, max_new_tokens=32, temperature=1):
    try:
        if "longalign" in model_name.lower() and max_new_tokens == 32:
            max_new_tokens = 128
        prompt, prompt_len = set_prompt(prompt, maxlen)
        history = []
        if "internlm" in model_name or "chatglm" in model_name or "longalign-6b" in model_name:
            response, history = model.chat(tokenizer, prompt, history=history, max_new_tokens=max_new_tokens, temperature=temperature, num_beams=1, do_sample=False)
            return response, prompt_len
        elif "baichuan" in model_name:
            messages = [{"content": prompt, "role": "user"}]
            model.generation_config = GenerationConfig.from_pretrained(model2path["baichuan2-7b-4k"], temperature=temperature, max_new_tokens=max_new_tokens, num_beams=1, do_sample=False)
            response = model.chat(tokenizer, messages)
            return response, prompt_len
        elif "llama3" in model_name:
            messages = [{"role": "user", "content": prompt}]
            input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            outputs = model.generate(input_ids, eos_token_id=terminators, max_new_tokens=max_new_tokens, temperature=temperature, num_beams=1, do_sample=False)
            response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
            return response, prompt_len
        elif "glm-4" in model_name or "glm3-turbo-128k" in model_name or "gpt" in model_name:
            response = call_api(prompt, model_name, max_new_tokens)
            return response, prompt_len
        elif "qwen" in model_name:
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
            generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=max_new_tokens, num_beams=1, do_sample=False, temperature=1.0)
            response = tokenizer.batch_decode([output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)], skip_special_tokens=True)[0]
            return response, prompt_len
        elif "llama" in model_name:
            input = tokenizer(f"[INST]{prompt}[/INST]", truncation=False, return_tensors="pt").to(model.device)
        elif "vicuna" in model_name:
            from fastchat.model import get_conversation_template
            conv = get_conversation_template("vicuna")
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            input = tokenizer(conv.get_prompt(), truncation=False, return_tensors="pt").to(model.device)
        context_length = input.input_ids.shape[-1]
        output = model.generate(**input, max_new_tokens=max_new_tokens, num_beams=1, do_sample=False, temperature=temperature)
        response = tokenizer.decode(output[0][context_length:], skip_special_tokens=True).strip()
    except Exception as e:
        print(f"An error occurred: {e}")
        time.sleep(1)
        return None, None
    return response, prompt_len



def setup_logger(logger, filename='log'):
    logger.setLevel(logging.DEBUG)
    if logger.hasHandlers():
        logger.handlers.clear()
    formatter = logging.Formatter(fmt="[%(asctime)s][%(levelname)s] - %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(os.path.join(log_path, filename))
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)

def print_args(args):
    logger.info(f"{'*' * 30} CONFIGURATION {'*' * 30}")
    for key, val in sorted(vars(args).items()):
        keystr = f"{key}{' ' * (30 - len(key))}"
        logger.info(f"{keystr} --> {val}")
    logger.info(f"LongRAG model used: {args.lrag_model}")
    logger.info(f"{'*' * 30} CONFIGURATION {'*' * 30}")

def search_q(question):
    doc_len = {}
    raw_pred = ""
    if args.raw_pred:
        raw_pred = search_cache_and_predict(raw_pred, f'{log_path}/raw_pred.json', 'raw_pred', question, model_name, model, tokenizer, lambda: create_prompt(question), maxlen)

    retriever, match_id = vector_search(question)
    rerank, match_id = sort_section(question, retriever, match_id)

    filter_output = []
    extractor_output = []
    fil_pred = ext_pred = ext_fil_pred = rb_pred = rl_pred = ''

    if args.fil:
        fil_pred = load_cache(f'{log_path}/fil_pred.json', 'fil_pred', question, doc_len, 'Fil')
        if not fil_pred:
            filter_output = filter(question, rerank)
            fil_pred = search_cache_and_predict(fil_pred, f'{log_path}/fil_pred.json', 'fil_pred', question, model_name, model, tokenizer, lambda: create_prompt(''.join(filter_output), question), maxlen, doc_len, 'Fil')
    
    if args.ext:
        ext_pred = load_cache(f'{log_path}/ext_pred.json', 'ext_pred', question, doc_len, 'Ext')
        if not ext_pred:
            extractor_output = extractor(question, rerank, match_id)
            ext_pred = search_cache_and_predict(ext_pred, f'{log_path}/ext_pred.json', 'ext_pred', question, model_name, model, tokenizer, lambda: create_prompt(''.join(rerank + extractor_output), question), maxlen, doc_len, 'Ext')

    if args.ext_fil:
        ext_fil_pred = load_cache(f'{log_path}/ext_fil_pred.json', 'ext_fil_pred', question, doc_len, 'E&F')
        if not ext_fil_pred:
            if not filter_output:
                filter_output = filter(question, rerank)
            if not extractor_output:
                extractor_output = extractor(question, rerank, match_id)
            ext_fil_pred = search_cache_and_predict(ext_fil_pred, f'{log_path}/ext_fil_pred.json', 'ext_fil_pred', question, model_name, model, tokenizer, lambda: create_prompt(''.join(filter_output + extractor_output), question), maxlen, doc_len, 'E&F')
    
    if args.rb:
        rb_pred = load_cache(f'{log_path}/rb_pred.json', 'rb_pred', question, doc_len, 'R&B')
        if not rb_pred:
            rb_pred = search_cache_and_predict(rb_pred, f'{log_path}/rb_pred.json', 'rb_pred', question, model_name, model, tokenizer, lambda: create_prompt(''.join(rerank), question), maxlen, doc_len, 'R&B')
    
    if args.rl:
        rl_pred = load_cache(f'{log_path}/rl_pred.json', 'rl_pred', question, doc_len, 'R&L')
        if not rl_pred:
            rl_pred = search_cache_and_predict(rl_pred, f'{log_path}/rl_pred.json', 'rl_pred', question, model_name, model, tokenizer, lambda: create_prompt(''.join(s2l_doc(rerank, match_id, maxlen)[0]), question), maxlen, doc_len, 'R&L')
    
    return question, retriever, rerank, raw_pred, rb_pred, ext_pred, fil_pred, rl_pred, ext_fil_pred, doc_len

def load_cache(cache_path, pred_key, question, doc_len=None, doc_key=None):
    if os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                data = json.loads(line)
                if data['question'] == question:
                    pred_result = data[pred_key]
                    if doc_len is not None and doc_key is not None:
                        doc_len[doc_key] = data["input_len"]
                    return pred_result
    return ''

def search_cache_and_predict(pred_result, cache_path, pred_key, question, model_name, model, tokenizer, create_prompt_func, maxlen, doc_len=None, doc_key=None):
    if not pred_result:
        query = create_prompt_func()
        pred_result, input_len = pred(model_name, model, tokenizer, query, maxlen)
        with open(cache_path, 'a', encoding='utf-8') as f:
            json.dump({'question': question, pred_key: pred_result, "input_len": input_len}, f, ensure_ascii=False)
            f.write('\n')
        if doc_len is not None and doc_key is not None:
            doc_len[doc_key] = input_len
    return pred_result

def s2l_doc(rerank, match_id, maxlen):
    unique_raw_id = []
    contents = []
    s2l_index = {}
    section_index = [id_to_rawid[str(i)] for i in match_id]
    for index, id in enumerate(section_index):
        data = raw_data[id]
        text = data["paragraph_text"]
        if id in unique_raw_id and get_word_len(text) < maxlen:
            continue
        if get_word_len(text) >= maxlen:
            content = rerank[index]
        else:
            unique_raw_id.append(id)
            content = text
        s2l_index[len(contents)] = [i for i, v in enumerate(section_index) if v == section_index[index]]
        contents.append(content)
    return contents, s2l_index


def filter(question,rank_docs): 
    
    content="\n".join(rank_docs)
    query=f"{content}\n\nPlease combine the above information and give your thinking process for the following question:{question}."
    think_pro,_=pred(lrag_model_name, lrag_model, lrag_tokenizer, query,lrag_maxlen,1000)
    selected = []

    prompts=[f"""Given an article:{d}\nQuestion: {question}.\nThought process:{think_pro}.\nYour task is to use the thought process provided to decide whether you need to cite the article to answer this question. If you need to cite the article, set the status value to True. If not, set the status value to False. Please output the response in the following json format: {{"status": "{{the value of status}}"}}""" for d in rank_docs]
    pool = ThreadPool(processes=args.MaxClients)
    all_responses=pool.starmap(pred, [(lrag_model_name,lrag_model, lrag_tokenizer,prompt,lrag_maxlen,32) for prompt in prompts])

    for i,r in enumerate(all_responses):
        try:    
            result=json.loads(r[0])
            res=result["status"] 
            if len(all_responses)!=len(rank_docs):
                break     
            if res.lower()=="true":
                selected.append(rank_docs[i])
        except:
            match=re.search("True|true",r[0])
            if match:
                selected.append(rank_docs[i])
    if len(selected)==0:
        selected=rank_docs
    return selected


def r2long_unique(rerank, match_id):
    unique_raw_id = list(set(id_to_rawid[str(i)] for i in match_id))
    section_index = [id_to_rawid[str(i)] for i in match_id]
    contents = [''.join(rerank[i] for i in range(len(section_index)) if section_index[i] == uid) for uid in unique_raw_id]
    return contents, unique_raw_id

def extractor(question, docs, match_id):
    long_docs = s2l_doc(docs, match_id, lrag_maxlen)[0]
    content = ''.join(long_docs)
    query = f"{content}.\n\nBased on the above background, please output the information you need to cite to answer the question below.\n{question}"
    response = pred(lrag_model_name, lrag_model, lrag_tokenizer, query, lrag_maxlen, 1000)[0]
    # logger.info(f"cite_passage responses: {all_responses}")
    return [response]


def vector_search(question):
    feature = emb_model.encode([question])
    distance, match_id = vector.search(feature, args.top_k1)
    content = [chunk_data[int(i)] for i in match_id[0]]
    return content, list(match_id[0])

def sort_section(question, section, match_id):
    q = [question] * len(section)
    features = cross_tokenizer(q, section, padding=True, truncation=True, return_tensors="pt").to(device)
    cross_model.eval()
    with torch.no_grad():
        scores = cross_model(**features).logits.squeeze(dim=1)
    sort_scores = torch.argsort(scores, dim=0, descending=True).cpu()
    result = [section[sort_scores[i].item()] for i in range(args.top_k2)]
    match_id = [match_id[sort_scores[i].item()] for i in range(args.top_k2)]
    return result, match_id

def create_prompt(input, question):
    user_prompt = f"Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{input}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {question}\nAnswer:"
    return user_prompt


if __name__ == '__main__':
    seed_everything(42)
    index_path = f'{args.r_path}/{args.dataset}/vector.index' # Vector index path
    vector = faiss.read_index(index_path)
    with open(f'../data/corpus/raw/{args.dataset}.json', encoding='utf-8') as f:
        raw_data = json.load(f)
    with open(f'{args.r_path}/{args.dataset}/id_to_rawid.json', encoding='utf-8') as f:
        id_to_rawid = json.load(f)
    with open(f"{args.r_path}/{args.dataset}/chunks.json", "r") as fin:
        chunk_data = json.load(fin)

    now = datetime.now() 
    now_time = now.strftime("%Y-%m-%d-%H:%M:%S")
    log_path = args.log_path or f'./log/{args.r_path.split("/")[-1]}/{args.dataset}/{args.model}/{args.lrag_model or "base"}/{now_time}'
    os.makedirs(log_path, exist_ok=True)

    with open("../config/config.yaml", "r") as file:
        config = yaml.safe_load(file)
    
    model_name = args.model.lower()
    model2path = config["model_path"]
    maxlen = config["model_maxlen"][model_name]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    emb_model = SentenceTransformer(model2path["emb_model"]).to(device)
    cross_tokenizer = AutoTokenizer.from_pretrained(model2path["rerank_model"])
    cross_model = AutoModelForSequenceClassification.from_pretrained(model2path["rerank_model"]).to(device)
    model, tokenizer = load_model_and_tokenizer(model2path, model_name)
    
    if args.lrag_model:
        lrag_model_name = args.lrag_model.lower()
        lrag_maxlen = config["model_maxlen"][lrag_model_name]
        lrag_model, lrag_tokenizer = (model, tokenizer) if model_name == lrag_model_name else load_model_and_tokenizer(model2path, lrag_model_name)
    else:
        lrag_model_name, lrag_model, lrag_tokenizer, lrag_maxlen = (model_name, model, tokenizer, maxlen)
    set_prompt_tokenizer = AutoTokenizer.from_pretrained(model2path["chatglm3-6b-32k"], trust_remote_code=True)
    setup_logger(logger)
    print_args(args)

    questions, answer, raw_preds, rank_preds, ext_preds, fil_preds, longdoc_preds, ext_fil_preds, docs_len = [], [], [], [], [], [], [], [], []
    with open(f'../data/eval/{args.dataset}.json', encoding='utf-8') as f:
        qs_data = json.load(f)

    for d in qs_data:
        questions.append(d["question"])
        answer.append(d["answers"])

    for index, query in tqdm(enumerate(questions)):
        logger.info(f"Question: {query}")
        question, retriever, rerank, raw_pred, rb_pred, ext_pred, fil_pred, rl_pred, ext_fil_pred, doc_len = search_q(query)

        raw_preds.append(raw_pred)
        rank_preds.append(rb_pred)
        ext_preds.append(ext_pred)
        fil_preds.append(fil_pred)
        longdoc_preds.append(rl_pred)
        ext_fil_preds.append(ext_fil_pred)
        docs_len.append(doc_len)

    all_len1 = all_len2 = all_len3 = all_len4 = all_len5 = 0
    for dl in docs_len:
        all_len1 += dl.get('Ext', 0)
        all_len2 += dl.get('Fil', 0)
        all_len3 += dl.get('R&B', 0)
        all_len4 += dl.get('R&L', 0)
        all_len5 += dl.get('E&F', 0)

    doc_len_eval = {
        "Ext": all_len1 / len(docs_len),
        "Fil": all_len2 / len(docs_len),
        "R&B": all_len3 / len(docs_len),
        "R&L": all_len4 / len(docs_len),
        "E&F": all_len5 / len(docs_len)
    }
    
    
    F1 = {
        "raw_pre": F1_scorer(raw_preds, answer),
        "R&B": F1_scorer(rank_preds, answer),
        "Ext": F1_scorer(ext_preds, answer),
        "Fil": F1_scorer(fil_preds, answer),
        "R&L": F1_scorer(longdoc_preds, answer),
        "E&F": F1_scorer(ext_fil_preds, answer)
    }

    eval_result = {"F1": F1, "doc_len": doc_len_eval}
    with open(f"{log_path}/eval_result.json", "w") as fout:
        json.dump(eval_result, fout, ensure_ascii=False, indent=4)