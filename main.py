from transformers import AutoModelWithLMHead, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("tuner007/t5_abs_qa")
model = AutoModelWithLMHead.from_pretrained("tuner007/t5_abs_qa")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def get_answer(question, context):
    input_text = "context: %s <question for context: %s </s>" % (context,question)
    features = tokenizer([input_text], return_tensors='pt')
    out = model.generate(input_ids=features['input_ids'].to(device), attention_mask=features['attention_mask'].to(device))

    return tokenizer.decode(out[0])

context = "The purpose of this research is to explore the relationship between supply chain management strategy and chain management practices on supply chain performance. The main tools of data collection instrument used was a questionnaire which was administrated to a total sample of 200 managers are classified by job title and respondents are also classified by their job functions are corporate executive, purchasing, manufacturing/production, distribution/logistic, SCM, transportation, material, and operation from Malaysia manufacturing industry. The response rate was 62% while 51% was usable questionnaires. Sample selection was based on convenience sampling. The data were analyzed using mean, standard deviation and correlation between independent and dependent variables. The analyses involved statistical methods such as reliability and validity tests and multiple regressions. The finding showed that supply chain management practices have a significant relationship with supply chain performance statically. However, supply chain management strategy is a weak predictor of supply chain management performance"

question = "What did the finding showed?"
out = get_answer(question, context)
print(out)
# output: 'It is a hall of worship ruled by Odin.'