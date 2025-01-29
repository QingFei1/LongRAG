import json
import random
import argparse
from tqdm import tqdm
import os
from task import build_ext_instruction, build_fil_instruction, build_cot_instruction,build_rag_instruction,get_word_len
parser = argparse.ArgumentParser()
parser.add_argument('--per_task_num', type=int, default=200, help="Number of instructions to generate for each task in each dataset")
parser.add_argument('--min_res_tokens', type=int, default=20, help="Minimum number of response tokens") 
parser.add_argument('--long_ratio', type=int, default=0.2, help="Proportion of long data") 
parser.add_argument('--model', type=str, choices=["chatglm_turbo", "glm-4", "gpt-4-32k"], default='chatglm_turbo')
args = parser.parse_args()


def predata(data):
    for d in data:
        no_support = [c for c in d["contexts"] if not c["is_supporting"]]
        support = [c for c in d["contexts"] if c["is_supporting"]]

        if len(no_support) > 3:
            no_support = random.sample(no_support, random.randint(2, len(no_support)))

        d["contexts"] = support + no_support
        random.shuffle(d["contexts"])

    return data


if __name__ == "__main__":


    """
    Configure the threshold for long and short contexts for each dataset. For example, set 2wikimultihopqa to 1500, which means if the total length
    of all paragraphs exceeds 1500 tokens, it is considered a long context; otherwise, it is considered a short context.
    """
    dataset_config = {"2wikimultihopqa": 1500, "hotpotqa": 1500, "musique": 2500}


    instruction_builders = {
        "ext": build_ext_instruction,
        "fil": build_fil_instruction,
        "cot": build_cot_instruction,
        "rag": build_rag_instruction
    }
    
    all_instructions = []
    for name, min_token in dataset_config.items():   
        print(f"{'*'*10}Generating {name} instruction{'*'*10}")
        progress_bars = {task_type: tqdm(total=args.per_task_num, desc=f"{task_type} progress") for task_type in instruction_builders.keys()}

        with open(f"../data/train/raw/{name}/train.jsonl", "r") as fin:
            data = [json.loads(line) for line in fin]

        random.shuffle(data)

        task_instructions = []
        task_counters = {key: [0,0] for key in instruction_builders.keys()}
        for index, d in enumerate(data):
            question = d["question"]
            answer = d["answer"]
            content = "\n".join([c["title"] + c["paragraph_text"] for c in d["contexts"]])
            support = [c["title"] + c["paragraph_text"] for c in d["contexts"] if c["is_supporting"]]
            non_support = [c["title"] + c["paragraph_text"] for c in d["contexts"] if not c["is_supporting"]]

            for task_type, builder in instruction_builders.items():
                short_status=False
                if get_word_len(content)<min_token:
                    short_status=True
                    if task_counters[task_type][1] >= args.per_task_num*args.long_ratio:
                        break
                if task_counters[task_type][0] < args.per_task_num:
                    if task_type == "fil":
                        sup_flag=True
                        if task_counters[task_type][0]%2==0:
                            sup_flag=False                
                        instruction = builder(args.model, question, answer, support, non_support, sup_flag)
                    elif task_type=="rag":
                        instruction = builder(args.model, question, answer, content)
                    else:
                        instruction = builder(args.model, question, answer, content, support, args.min_res_tokens)

                    if instruction:
                        if short_status and task_counters[task_type][1] < args.per_task_num*args.long_ratio:
                            task_counters[task_type][1] += 1
                        task_instructions.append(instruction)
                        task_counters[task_type][0] += 1
                        progress_bars[task_type].update(1)               
                    break

            if all(task_counters[task][0] >= args.per_task_num for task in task_counters):
                break
        all_instructions.extend(task_instructions)
    for bar in progress_bars.values():
       bar.close()

    # Build longer context training data for ext and cot tasks
    with open(f"../data/train/raw/qasper/train.jsonl", "r") as fin:
        data = [json.loads(line) for line in fin]
    cot_instructions=[]
    ext_instructions=[]
    for index, d in enumerate(data):
        question = d["question"]
        answer = d["answer"]
        content = d["content"]
        support = d["support"]
        if d["answer"] and len(cot_instructions)<args.per_task_num//2:
            cot_instruction = build_cot_instruction(args.model, question, answer, content, support, args.min_res_tokens)
            if not cot_instruction:
                continue
            cot_instructions.append(cot_instruction)
        else:
            ext_instruction = build_ext_instruction(args.model, question, answer, content, support, args.min_res_tokens)
            if not ext_instruction:
                continue
            ext_instructions.append(ext_instruction)
        if len(cot_instructions)>=args.per_task_num//2 and len(ext_instructions)>=args.per_task_num//2:
            break
    all_instructions+=ext_instructions+cot_instructions
    random.shuffle(all_instructions)

    save_path = f"../data/train/processed"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(f"{save_path}/LRGinstruction.json", 'w') as fout:
        json.dump(all_instructions, fout, ensure_ascii=False)

