### I haven't runned this code because of I don't have GPU at that time
## therefore, no processed datasets for this finetune code

import math
import json
from torch.utils.data import DataLoader
from sentence_transformers import InputExample, LoggingHandler
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator

import logging
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)
logger = logging.getLogger(__name__)

train_batch_size = 4
num_epochs = 3

model_id = "Alibaba-NLP/gte-multilingual-reranker-base"
model_save_path = "gte-multilingual-reranker-base-new"
model = CrossEncoder(model_id, num_labels=1, trust_remote_code=True)

data_dir = "training_data"

train_data = list()
for d in json.loads(open(f"{data_dir}/train.json").read()):
    train_data.append(InputExample(texts=[d["sentence1"], d["sentence2"]], label=d["score"]))

dev_data = list()
for d in json.loads(open(f"{data_dir}/dev.json").read()):
    dev_data.append(InputExample(texts=[d["sentence1"], d["sentence2"]], label=d["score"]))

test_data = list()
for d in json.loads(open(f"{data_dir}/test.json").read()):
    test_data.append(InputExample(texts=[d["sentence1"], d["sentence2"]], label=d["score"]))

logger.info(f"Total Train Data: {len(train_data)}, Dev Data: {len(dev_data)}, Test Data: {len(test_data)}")


logger.info(f"Training the model")
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size)
evaluator = CECorrelationEvaluator.from_input_examples(dev_data, name="sts-dev")

warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)
logger.info(f"Warmup-steps: {warmup_steps}")

model.fit(
    train_dataloader=train_dataloader,
    evaluator=evaluator,
    epochs=num_epochs,
    warmup_steps=warmup_steps,
    output_path=model_save_path,
    use_amp=True
)

# eval
model = CrossEncoder(model_save_path, trust_remote_code=True)
evaluator = CECorrelationEvaluator.from_input_examples(test_data, name="sts-test")
logger.info(evaluator(model))