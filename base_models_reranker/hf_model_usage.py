# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch 

# tokenizer = AutoTokenizer.from_pretrained("wl-tookitaki/bge_reranker")
# model = AutoModelForSequenceClassification.from_pretrained("wl-tookitaki/bge_reranker")

# query = "Who wrote 'To Kill a Mockingbird'?"
# documents = [
#     "'To Kill a Mockingbird' is a novel by Harper Lee published in 1960. It was immediately successful, winning the Pulitzer Prize, and has become a classic of modern American literature.",
#     "The novel 'Moby-Dick' was written by Herman Melville and first published in 1851. It is considered a masterpiece of American literature and deals with complex themes of obsession, revenge, and the conflict between good and evil.",
#     "Harper Lee, an American novelist widely known for her novel 'To Kill a Mockingbird', was born in 1926 in Monroeville, Alabama. She received the Pulitzer Prize for Fiction in 1961.",
#     "Jane Austen was an English novelist known primarily for her six major novels, which interpret, critique and comment upon the British landed gentry at the end of the 18th century.",
#     "The 'Harry Potter' series, which consists of seven fantasy novels written by British author J.K. Rowling, is among the most popular and critically acclaimed books of the modern era.",
#     "'The Great Gatsby', a novel written by American author F. Scott Fitzgerald, was published in 1925. The story is set in the Jazz Age and follows the life of millionaire Jay Gatsby and his pursuit of Daisy Buchanan."
# ]
# # Encode each document with the query
# inputs = [tokenizer(query, doc, return_tensors="pt", truncation=True, padding=True, max_length=512) for doc in documents]

# # Collect scores for each document
# scores = []
# with torch.no_grad():
#     for input in inputs:
#         output = model(**input)
#         scores.append(output.logits)

# # Print scores
# for i, score in enumerate(scores):
#     relevance_score = score[0].item()  # 假设模型输出的第二个logit表示相关性评分
#     print(f"Document {i+1} Relevance Score: {relevance_score}")

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 加載模型和tokenizer
model_name_or_path = "Alibaba-NLP/gte-multilingual-reranker-base"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name_or_path, trust_remote_code=True,
    torch_dtype=torch.float16
)
model.eval()

# 定義查詢和文檔
query = "Who wrote 'To Kill a Mockingbird'?"
documents = [
    "'To Kill a Mockingbird' is a novel by Harper Lee published in 1960. It was immediately successful, winning the Pulitzer Prize, and has become a classic of modern American literature.",
    "The novel 'Moby-Dick' was written by Herman Melville and first published in 1851. It is considered a masterpiece of American literature and deals with complex themes of obsession, revenge, and the conflict between good and evil.",
    "Harper Lee, an American novelist widely known for her novel 'To Kill a Mockingbird', was born in 1926 in Monroeville, Alabama. She received the Pulitzer Prize for Fiction in 1961.",
    "Jane Austen was an English novelist known primarily for her six major novels, which interpret, critique and comment upon the British landed gentry at the end of the 18th century.",
    "The 'Harry Potter' series, which consists of seven fantasy novels written by British author J.K. Rowling, is among the most popular and critically acclaimed books of the modern era.",
    "'The Great Gatsby', a novel written by American author F. Scott Fitzgerald, was published in 1925. The story is set in the Jazz Age and follows the life of millionaire Jay Gatsby and his pursuit of Daisy Buchanan."
]

# 構建查詢和文檔對
pairs = [[query, doc] for doc in documents]

# 對查詢和文檔對進行編碼
with torch.no_grad():
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
    scores = model(**inputs, return_dict=True).logits.view(-1, ).float()

# 根據分數對文檔進行排序
sorted_documents = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

# 輸出排序結果
for doc, score in sorted_documents:
    print(f"Score: {score:.4f}\nDocument: {doc}\n")