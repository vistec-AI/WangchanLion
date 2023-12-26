from datasets import load_dataset , concatenate_datasets
from helper_functions import *
import torch
from trl.trainer import ConstantLengthDataset
from omegaconf import OmegaConf
import random
import glob
import json
from torch.utils.data import IterableDataset
from datasets import disable_caching
disable_caching()

def split_dataset(args, dataset):
    if args.streaming:
        print("Loading the dataset in streaming mode")
        valid_data = dataset.take(args.size_valid_set)
        train_data = dataset.skip(args.size_valid_set)
    else:
        dataset = dataset.train_test_split(test_size=0.00001, seed=None)
        train_data = dataset["train"]
        valid_data = dataset["test"]
    return train_data, valid_data


def deterministic_random(seed: int) -> random.Random:
    return random.Random(seed)

def remove_not_relate_to_thaiEn_dolphin(example):
   a = example["input"]
   fullstring = example["input"].lower()
   substring1 = "language" 
   substring2 = "translate"

   if substring1 in fullstring or substring2 in fullstring :
        example["has_tran"] = 1
   else:
        example["has_tran"] = 0
   
   ins =  example["instruction"] +" "+ example["input"] 
   example["Answer"] = example["output"].strip()
   example["Context"] = ""
   example["Instruction"] = ins.strip()
   
   return example 

def re_formate_iappQA(example):

   combine = "พื้นหลัง: " + example["context"].strip() +"\nคำถาม: " + example["question"].strip()
   example["Answer"] = example["answers"]["text"][0].strip()
   example["Context"] = ""
   example["Instruction"] =  combine.strip()
   
   return example

def re_formate_thaisum(example):

   combine = "เนื้อหาข่าว (news): " + example["body"].strip()+ "\nจากเนื้อหาข่าว จงสรุปเนื้อหาให้กระชับ เข้าใจง่าย"
   example["Answer"] = example["summary"].strip() 
   example["Context"] = ""
   example["Instruction"] = combine.strip()
   
   return example

def re_formate_xlsum(example):
   combine = "เนื้อหา บทความ (article): " + example["text"].strip()+ "\nจากเนื้อหาบทความที่กล่าวก่อนหน้านี้ จงสรุปบทความโดยมีเนื้อหาที่สั้นและเข้าใจง่าย"
   example["Answer"] = example["summary"].strip() 
   example["Context"] = ""
   example["Instruction"] = combine.strip()
   
   return example

def re_formate_scb_then(example):
   example["Answer"] = example['translation']['en'].strip()
   example["Context"] = ""
   example["Instruction"] = "Translate Thai to English. จงแปลภาษาไทยเป็นอังกฤษ\n" + example['translation']['th'].strip()
   
   return example

def re_formate_scb_enth(example):
   example["Answer"] = example['translation']['th']
   example["Context"] = ""
   example["Instruction"] = "Translate English to Thai. จงแปลภาษาอังกฤษเป็นภาษาไทย\n" + example['translation']['en'].strip()
   
   return example

def re_formate_xp3(example):
   example["Answer"] = example["targets"].strip()
   example["Context"] = ""
   example["Instruction"] = example["inputs"].strip()

   return example


def read_file():

    path = "/your path file/*"
    all_file = []
    j = 0
    for file in glob.glob(path):
        j = j + 1
        try:
            with open(file) as f:
                    data = [json.loads(line) for line in f]
                    
                    all_file.append(file)
        except Exception as e:
            continue
    return all_file

def re_formate_han(example):

   example["Answer"] = example["a"].strip()
   example["Context"] = ""
   example["Instruction"] =  example["q"].strip()
   
   return example

def re_formate_platypus(example):

   if example["input"]  != "":
      comb = example["instruction"] +"\n"+ example["input"]
   else:
      comb = example["instruction"]

   example["Answer"] = example["output"]
   example["Context"] = ""
   example["Instruction"] =  comb.strip()
   return example


def load_all_dataset(args, split_train_test=True):
    dataset = load_dataset(
           "pythainlp/final_training_set_v1",
           split="train",
           num_proc=8 if not args.streaming else None,
           streaming = args.streaming
           )
    dataset = dataset.map(add_prefix)
    dataset = dataset.filter(lambda x: x["bot_morethan_one"] == 2)
    dataset = dataset.remove_columns([ "has_context","bot_morethan_one" , "text" , "metadata", "nb_token" ])

    dataset_dolphin = load_dataset("ehartford/dolphin",data_files="flan1m-alpaca-uncensored.jsonl" ,split="train")
    random_test_ids = deterministic_random(42).sample(range(len(dataset_dolphin)), k=10000)
    dataset_dolphin = dataset_dolphin.select(random_test_ids)
    dataset_dolphin = dataset_dolphin.map(remove_not_relate_to_thaiEn_dolphin)
    dataset_dolphin = dataset_dolphin.filter(lambda x: x["has_tran"] == 0)
    dataset_dolphin = dataset_dolphin.remove_columns(['instruction', 'input', 'output', 'has_tran'])


    dataset_iapp = load_dataset("iapp_wiki_qa_squad",split="train")
    dataset_iapp = dataset_iapp.map(re_formate_iappQA)
    dataset_iapp = dataset_iapp.remove_columns(['question_id', 'article_id', 'title', 'context', 'question', 'answers'])

    dataset_thaisum = load_dataset("thaisum",split="train")
    random_test_ids = deterministic_random(42).sample(range(len(dataset_thaisum)), k=5000)
    dataset_thaisum = dataset_thaisum.select(random_test_ids)
    dataset_thaisum  = dataset_thaisum.map(re_formate_thaisum)
    dataset_thaisum = dataset_thaisum.remove_columns(['title', 'body', 'summary', 'type', 'tags', 'url'])

    dataset_xlsum = load_dataset("csebuetnlp/xlsum" , "thai",split="train")
    random_test_ids = deterministic_random(42).sample(range(len(dataset_xlsum)), k=5000)
    dataset_xlsum = dataset_xlsum.select(random_test_ids)
    dataset_xlsum = dataset_xlsum.map(re_formate_xlsum)
    dataset_xlsum = dataset_xlsum.remove_columns(['id', 'url', 'title', 'summary', 'text'])

    
    dataset_enth = load_dataset("scb_mt_enth_2020" , "enth",split="train")

    random_test_ids = deterministic_random(42).sample(range(len(dataset_enth)), k=10000)
    dataset_enth = dataset_enth.select(random_test_ids)
    dataset_then = dataset_enth

    dataset_enth = dataset_enth.map(re_formate_scb_enth)
    dataset_enth = dataset_enth.remove_columns(['translation', 'subdataset'])

    dataset_then  = dataset_then.map(re_formate_scb_then)
    dataset_then = dataset_then.remove_columns(['translation', 'subdataset'])


    dataset_han = load_dataset("csv", data_files="/workspace/sealion/examples/pythainlp-han_dataset - pythainlp-han_dataset.csv")
    dataset_han = dataset_han.map(re_formate_han).remove_columns(['q', 'a'])

    all_file = read_file()
    dataset_ = load_dataset("json", data_files=all_file, split="train")
    dataset_1 = dataset_.filter(lambda example: example["split"] == "train")


    dataset_en_th = dataset_1.filter(lambda example: example["config"] in ['en_th']) # ['en_th', 'th', 'thai' ]
    random_test_ids = deterministic_random(42).sample(range(len(dataset_en_th)), k=40000)
    dataset_en_th = dataset_en_th.select(random_test_ids)
    dataset_en_th = dataset_en_th.map(re_formate_xp3)
    dataset_en_th = dataset_en_th.remove_columns(['inputs', 'targets', 'language', 'split', 'template', 'dataset', 'config'])


    dataset_thai = dataset_1.filter(lambda example: example["config"] in ['thai'])
    dataset_thai = dataset_thai.map(re_formate_xp3)
    dataset_thai = dataset_thai.remove_columns(['inputs', 'targets', 'language', 'split', 'template', 'dataset', 'config'])

    dataset_Platypus = load_dataset("garage-bAInd/Open-Platypus", split="train")
    dataset_Platypus = dataset_Platypus.filter(lambda example: example["data_source"] != "scienceqa" and example["data_source"] != "reclor") #due to copyright
    dataset_Platypus = dataset_Platypus.map(re_formate_platypus)
    dataset_Platypus = dataset_Platypus.remove_columns(['input', 'output', 'instruction', 'data_source'])


    dataset = concatenate_datasets([
    dataset
    ,dataset_dolphin
    , dataset_iapp
    , dataset_thaisum
    , dataset_xlsum
    , dataset_enth
    , dataset_then
    ,dataset_en_th
    ,dataset_thai
    ,dataset_han["train"]
    ,dataset_Platypus
    ])


    dataset = dataset.map(prepare_dolly_text)

    if split_train_test:
        return split_dataset(args, dataset)
    else:
        return dataset




def add_prefix(example):
    a = example["text"].split("<bot>:")

    example["bot_morethan_one"] = len(a)
    example["has_context"] = 1 if "<context>:" in example["text"] else 0

    v = example["text"]

    # Find the indices of the tags
    context_index = v.find("<context>:")
    human_index = v.find("<human>:")
    bot_index = v.find("<bot>:")

    context = v[context_index:human_index].replace("<context>:","").strip() 
    human = v[human_index:bot_index].replace("<human>:","").strip()
    bot = v[bot_index:].replace("<bot>:","").strip()

    combined = ""
    if context != "":
       combined = context +"\n" + human
    else:
       combined = human
       
    example["Context"] = ""
    example["Instruction"] = combined.strip()
    example["Answer"] = bot.strip()

    return example


def prepare_dolly_text(example):

    if example["Answer"] != "":
        text = example["Instruction"] +"\n"+ example["Answer"]
    else:
        text = example["Instruction"] 

    example["text"] = text
    return example


def create_datasets(tokenizer, conf):
    train_data_list = []
    valid_data_list = []

    train_data, valid_data = load_all_dataset(conf, tokenizer)

    if conf.streaming:
        print("Loading the dataset in streaming mode")
        train_data = train_data.shuffle(buffer_size=conf.shuffle_buffer, seed=None)
    else:
        train_data = train_data.shuffle(seed=None)
        print(
            f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}"
        )

    chars_per_token = chars_token_ratio(train_data, tokenizer)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        dataset_text_field="text",
        infinite=True,
        seq_length=conf.seq_length,
        chars_per_token=chars_per_token,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        dataset_text_field="text",
        infinite=False,
        seq_length=conf.seq_length,
        chars_per_token=chars_per_token,
    )
    return train_dataset, valid_dataset
