# Please use transformers==4.34.1
from transformers import AutoModelForCausalLM, AutoTokenizer

generation_kwargs = {
    "do_sample": False,  # set to true if temperature is not 0
    "temperature": None,
    "max_new_tokens": 256,
    "top_k": 50,
    "top_p": 0.7,
    "repetition_penalty": 1.2,
}

tokenizer = AutoTokenizer.from_pretrained(
    "airesearch/WangchanLion7B", trust_remote_code=True
)
generation_kwargs["eos_token_id"] = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained(
    "airesearch/WangchanLion7B", trust_remote_code=True
)

BACKGROUND = "พื้นหลัง:"
QUESTION = "คำถาม:"
ANSWER  = "ตอบ:"
question_prompt = "แนะนำวิธีการลดน้ำหนัก"

full_prompt = QUESTION + question_prompt + '\n\n' + ANSWER
tokens = tokenizer(full_prompt, return_tensors="pt")

# Remove unneeded kwargs
if generation_kwargs["do_sample"] == False:
    generation_kwargs.pop("temperature")
    generation_kwargs.pop("top_k")
    generation_kwargs.pop("top_p")

output = model.generate(
    tokens["input_ids"],
    **generation_kwargs,
)
print(tokenizer.decode(output[0], skip_special_tokens=True))
