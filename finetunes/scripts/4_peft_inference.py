from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch

model_name = "xxx"
model = AutoPeftModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = model.to("cuda")
model.eval()

inputs = tokenizer("Preheat the oven to 350 degrees and place the cookie though", return_tensors="pt")

outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=50)
print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0])


### For other tasks that aren’t explicitly supported with an AutoPeftModelFor class：
# - such as automatic speech recognition
# - you can still use the base AutoPeftModel class to load a model for the task.

# from peft import AutoPeftModel

# model = AutoPeftModel.from_pretrained("adasfsadgag")