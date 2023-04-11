import torch

# GPTJ 6B - 24 GB
from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")

#GPT2 548 MB / too large context?
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
# model = GPT2LMHeadModel.from_pretrained("gpt2")
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def get_answer_variant(question, context):
    print("Running On : " + str(device))
    input_text = "context: %s <question for context: %s </s>" % (context,question)
    features = tokenizer([input_text], return_tensors='pt')
    out = model.generate(input_ids=features['input_ids'].to(device), attention_mask=features['attention_mask'].to(device))

    return tokenizer.decode(out[0])
