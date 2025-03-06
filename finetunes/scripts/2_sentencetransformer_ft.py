import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

from datasets import load_dataset
from sentence_transformers import CrossEncoder, SentencesDataset, InputExample, losses
from torch.utils.data import DataLoader

class bge():
    def __init__(self):
        self.dataset = load_dataset('jinaai/nq-reranking-en')
        self.model = CrossEncoder('BAAI/bge-reranker-v2-m3')

    def get_act_data(self):
        queries = self.dataset['eval']['query']
        positives = self.dataset['eval']['positive']
        return queries, positives

    def get_neg_data(self):
        negatives = self.dataset['eval']['negative']  # Assuming negative examples are stored as lists in a string format
        return negatives

    def build_train_data(self, queries, positives, negatives, train_size):
        train_data = []
        for i in range(train_size):
            for pos in positives[i]:
                train_data.append(InputExample(texts=[queries[i], pos], label=1.0))
            for neg in negatives[i]:
                train_data.append(InputExample(texts=[queries[i], neg], label=0.0))
        return train_data

    def build_evaluation_data(self, queries, positives, negatives, train_size, eval_size):
        eval_data = []
        for i in range(train_size, len(queries)):
            for pos in positives[i]:
                eval_data.append(InputExample(texts=[queries[i], pos], label=1.0))
            for neg in negatives[i]:
                eval_data.append(InputExample(texts=[queries[i], neg], label=0.0))
        return eval_data

    def custom_evaluation(self, model, eval_data):
        correct = 0
        total = 0
        for example in eval_data:
            prediction = model.predict([example.texts])[0]
            if (prediction >= 0.5 and example.label == 1.0) or (prediction < 0.5 and example.label == 0.0):
                correct += 1
            total += 1
        accuracy = correct / total
        print(f'Custom Evaluation Accuracy: {accuracy:.4f}')
        return accuracy

    def callback(self, score, epoch, steps):
        print('score:{}, epoch:{}, steps:{}'.format(score, epoch, steps))

    def train(self):
        queries, positives = self.get_act_data()
        negatives = self.get_neg_data()

        train_size = int(len(queries) * 0.8)
        eval_size = len(queries) - train_size
        train_data = self.build_train_data(queries, positives, negatives, train_size)

        eval_data = self.build_evaluation_data(queries, positives, negatives, train_size, eval_size)

        train_dataset = SentencesDataset(train_data, self.model)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=4)
        train_loss = losses.CosineSimilarityLoss(self.model)

        self.model.fit(
            train_dataloader=train_dataloader,
            epochs=1,
            warmup_steps=100,
            evaluator=lambda model, output_path, epoch, steps: self.custom_evaluation(model, eval_data),
            evaluation_steps=100,
            output_path='ft-bge-reranker',
            save_best_model=True,
            callback=self.callback
        )

trainer = bge()
trainer.train()


# from sentence_transformers import CrossEncoder

# model = CrossEncoder('ft-bge-reranker')

# query = "Who wrote 'To Kill a Mockingbird'?"
# documents = [
#     "'To Kill a Mockingbird' is a novel by Harper Lee published in 1960. It was immediately successful, winning the Pulitzer Prize, and has become a classic of modern American literature.",
#     "The novel 'Moby-Dick' was written by Herman Melville and first published in 1851. It is considered a masterpiece of American literature and deals with complex themes of obsession, revenge, and the conflict between good and evil.",
#     "Harper Lee, an American novelist widely known for her novel 'To Kill a Mockingbird', was born in 1926 in Monroeville, Alabama. She received the Pulitzer Prize for Fiction in 1961.",
#     "Jane Austen was an English novelist known primarily for her six major novels, which interpret, critique and comment upon the British landed gentry at the end of the 18th century.",
#     "The 'Harry Potter' series, which consists of seven fantasy novels written by British author J.K. Rowling, is among the most popular and critically acclaimed books of the modern era.",
#     "'The Great Gatsby', a novel written by American author F. Scott Fitzgerald, was published in 1925. The story is set in the Jazz Age and follows the life of millionaire Jay Gatsby and his pursuit of Daisy Buchanan."
# ]

# scores = model.predict([(query, doc) for doc in documents])
# print(scores)

# best_answer = documents[scores.argmax()]
# print(f"Best answer: {best_answer}")