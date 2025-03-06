import transformers
from transformers import AutoTokenizer, TrainingArguments, get_cosine_schedule_with_warmup, AdamW
from retrievals import AutoModelForRanking, RerankCollator, RerankTrainDataset, RerankTrainer
transformers.logging.set_verbosity_error()

import os
os.environ["WANDB_DISABLED"] = "true"

model_name_or_path: str = "BAAI/bge-reranker-v2-m3"
max_length: int = 256
learning_rate: float = 2e-5
batch_size: int = 2
epochs: int = 1

train_dataset = RerankTrainDataset('jinaai/nq-reranking-en', positive_key='positive', negative_key='negative', dataset_split='dev')
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
model = AutoModelForRanking.from_pretrained(model_name_or_path)
optimizer = AdamW(model.parameters(), lr=learning_rate)
num_train_steps = int(len(train_dataset) / batch_size * epochs)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0.05 * num_train_steps, num_training_steps=num_train_steps)

training_args = TrainingArguments(
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    num_train_epochs=epochs,
    output_dir = './checkpoints',
    remove_unused_columns=False,
    logging_steps=100,
)
trainer = RerankTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=RerankCollator(tokenizer, max_length=max_length),
)
trainer.optimizer = optimizer
trainer.scheduler = scheduler
trainer.train()