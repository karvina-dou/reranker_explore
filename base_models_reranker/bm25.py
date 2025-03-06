import math
from collections import Counter

class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.doc_lengths = [len(doc) for doc in corpus]
        self.avg_doc_length = sum(self.doc_lengths)/len(self.doc_lengths)
        self.doc_count = len(corpus)
        self.doc_term_freqs = [Counter(doc.split()) for doc in corpus]
        self.inverted_index = self.build_inverted_index()
    
    def build_inverted_index(self):
        inverted_index = {} # dictionary
        for doc_id, doc_term_freq in enumerate(self.doc_term_freqs): # enumerate will create index
            for term, freq in doc_term_freq.items():
                if term not in inverted_index:
                    inverted_index[term] = []
                inverted_index[term].append((doc_id, freq))
        return inverted_index
    
    def idf(self, term):
        doc_freq = len(self.inverted_index.get(term, []))
        if doc_freq ==0:
            return 0
        return math.log(self.doc_count - doc_freq + 0.5) / (doc_freq + 0.5)
    
    def bm25_score(self, query_terms, doc_id):
        score = 0
        doc_length = self.doc_lengths[doc_id]
        for term in query_terms:
            tf = self.doc_term_freqs[doc_id].get(term, 0)
            idf = self.idf(term)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
            score += idf * (numerator / denominator)
        return score
    
    def rank_documents(self, query):
        query_terms = query.split()
        scores = [(doc_id, self.bm25_score(query_terms,doc_id)) for doc_id in range(self.doc_count)]
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        return sorted_scores
    
# corpus = [
#     "The quick brown fox jumps over the lazy dog",
#     "A quick brown dog outpaces a swift fox",
#     "The dog is lazy but the fox is swift",
#     "Lazy dogs and swift foxes"
# ]

corpus = [
    "'To Kill a Mockingbird' is a novel by Harper Lee published in 1960. It was immediately successful, winning the Pulitzer Prize, and has become a classic of modern American literature.",
    "The novel 'Moby-Dick' was written by Herman Melville and first published in 1851. It is considered a masterpiece of American literature and deals with complex themes of obsession, revenge, and the conflict between good and evil.",
    "Harper Lee, an American novelist widely known for her novel 'To Kill a Mockingbird', was born in 1926 in Monroeville, Alabama. She received the Pulitzer Prize for Fiction in 1961.",
    "Jane Austen was an English novelist known primarily for her six major novels, which interpret, critique and comment upon the British landed gentry at the end of the 18th century.",
    "The 'Harry Potter' series, which consists of seven fantasy novels written by British author J.K. Rowling, is among the most popular and critically acclaimed books of the modern era.",
    "'The Great Gatsby', a novel written by American author F. Scott Fitzgerald, was published in 1925. The story is set in the Jazz Age and follows the life of millionaire Jay Gatsby and his pursuit of Daisy Buchanan."
]

bm25 = BM25(corpus)
# query = "quick brown dog"
query = "Who wrote 'To Kill a Mockingbird'?"
result = bm25.rank_documents(query)

print("BM25 Scores for the query '{}':".format(query))
for doc_id, score in result:
    print("Document {}: {}".format(doc_id, score))