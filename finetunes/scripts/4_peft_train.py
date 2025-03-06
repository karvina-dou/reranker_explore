import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# LoRA
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=4,
    lora_alpha=16,
    lora_dropout=0.1
)

model = get_peft_model(model, lora_config)
print(model.print_trainable_parameters())


dataset = load_dataset("imdb")
train_dataset = dataset['train'].shuffle().select(range(1000))
test_dataset = dataset['test'].shuffle().select(range(100))

def preprocess(examples):
    processed = tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)
    return processed

train_dataset = train_dataset.map(preprocess, batched=True)
test_dataset = test_dataset.map(preprocess, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
)

# define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)

trainer.train()
model.save_pretrained("./lora_model")
tokenizer.save_pretrained("./lora_model")

results = trainer.evaluate()
print(results)