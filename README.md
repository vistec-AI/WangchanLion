# WangchanLion

WangchanLion is an instruction-finetuned model based on SEA-LION, a pan-ASEAN pretrained LLM development led by AI Singapore. The finetuning of WangchanLion is a collaborative effort between VISTEC and AI Singapore. 

What WangchanLion offers:
- Transparent pretrained model: The development of SEA-LION is community-driven, with different ASEAN collaborators contributing pretraining datasets. The SEA-LION developers ensure that all datasets are safe and can be utilized without commercial restrictions. This transparency extends to the provision of pretraining code, ensuring anyone can replicate SEA-LION using the provided datasets.
- Transparent finetuning data: In the spirit of open science, we make the finetuning data for WangchanLion accessible to all. This commitment to openness empowers the community by providing complete visibility into the instruction finetuning data that shapes WangchanLion.
- Transparent finetuning code: The finetuning code for WangchanLion is readily available for distribution. By sharing our methods and processes, we invite others to learn from, build upon, and innovate alongside us.

## Model Sources
- Model:  [https://github.com/vistec-AI/WangchanLion](https://huggingface.co/airesearch/WangchanLion7B)
- Demo: [demo_WangchanLion.ipynb - Colaboratory](https://colab.research.google.com/drive/1y_7oOU3ZJI0h4chUrXFL3K4kelW_OI2G?usp=sharing#scrollTo=4yN3Bo6iAH2L)

# Use cases
## Direct Use
Intended to be used as an instruction-following model for reading comprehension, brainstorming, and creative writing.

## Downstream Use
The model can be finetuned for any typical instruction-following use cases.

## Out-of-Scope Use
We do not expect the models to perform well in math problems, reasoning, and factfulness.
 
## Bias, Risks, and Limitations
We noticed similar limitations to other finetuned instruction followers, such as math problems, reasoning, and factfulness. Even though the models do not perform on the level that we expect them to be abused, they do contain undesirable biases and toxicity and should be further optimized for your particular use cases.

## Recommendations
Users (both direct and downstream) should be made aware of the risks, biases, and limitations of the model. More information is needed for further recommendations.
 
# How to Get Started with the Model
Use the code [here](https://colab.research.google.com/drive/1y_7oOU3ZJI0h4chUrXFL3K4kelW_OI2G?usp=sharing#scrollTo=4yN3Bo6iAH2L) to get started with the model.

Or

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained( "airesearch/WangchanLion7B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "airesearch/WangchanLion7B", trust_remote_code=True,
    return_dict=True,
    load_in_8bit=True ,
    device_map="auto",
    torch_dtype=torch.float16,
    offload_folder="./",
    low_cpu_mem_usage=True,
)
def get_prompt(question: str,context: str = None) -> str:
    if context is not None:
      return """พื้นหลัง:\n\n{context}\n\nคำถาม:{question}\n\nตอบ:""".format(context=context, question=question)
    return """คำถาม:{question}\n\nตอบ:""".format(question=question)

question = "เกิดอะไรขึ้นที่เทียนอันเหมินตอนปี 1989"
full_prompt = get_prompt(question=question)
tokens = tokenizer(full_prompt, return_tensors="pt").to("cuda")
output = model.generate(
    input_ids=tokens['input_ids'],
    attention_mask=tokens['attention_mask'],
    max_new_tokens=256,
    early_stopping=True,
    top_k=50, top_p=0.95,
    do_sample=True,
    temperature=0.3,
    repetition_penalty = 1.2,
    eos_token_id = tokenizer.eos_token_id,
)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

# Training Details
## Training Data
Finetuning datasets are sourced from [LAION OIG chip2 and infill_dbpedia (Apache-2.0)](https://huggingface.co/datasets/laion/OIG), [DataBricks Dolly v2 (Apache-2.0)](https://github.com/databrickslabs/dolly), [OpenAI TL;DR (MIT)](https://github.com/openai/summarize-from-feedback), [Hello-SimpleAI HC3 (CC-BY SA)](https://huggingface.co/datasets/Hello-SimpleAI/HC3), [dolphin](https://huggingface.co/datasets/ehartford/dolphin), [iapp_wiki_qa_squad](https://huggingface.co/datasets/iapp_wiki_qa_squad) , [thaisum](https://huggingface.co/datasets/thaisum), [xlsum](https://huggingface.co/datasets/csebuetnlp/xlsum), [scb_mt_enth_2020](https://huggingface.co/datasets/scb_mt_enth_2020), [han dataset](https://huggingface.co/datasets/pythainlp/han-instruct-dataset-v1.0), [xp3x](https://huggingface.co/datasets/Muennighoff/xP3x) and [Open-Platypus](https://huggingface.co/datasets/garage-bAInd/Open-Platypus).
## Training regime
- QLoRA with 4 A100 (40GB)
  
## Training script
```python
accelerate launch supervised_fine_tuning.py --config_path="train_config.yaml"
```

 
# Evaluation
We performed human and machine evaluations on XQuAD zero-shot and one-shot settings:
## XQuAD
|      Model     | F1 (Zero-shot) | F1 (One-shot) |
|:--------------:|:--------------:|:-------------:|
| openthaigpt7B  |     27.3487      |    34.3104     |
| SeaLLM7B V2       |    16.1104       |  25.7399        |
| Typhoon-7b     |     34.46      |    **54.03**  |
| WangchanLion7B |   **45.8763**    |    49.9145      |

## iAPP Wiki QA 
|      Model     | F1 (Zero-shot) | F1 (One-shot) |
|:--------------:|:--------------:|:-------------:|
| openthaigpt7B  |       40.0614     |    46.6883    |
| SeaLLM7B       |       23.6425    |    28.9934    |
| WangchanLion7B |   **58.9051**  |  **62.9776**  |
