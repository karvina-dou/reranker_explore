# %% [markdown]
# ## Query Rewriting
# 
# Reformulates queries to be more specific and detailed.

# %%
from ollama import Client
from langchain.prompts import PromptTemplate

client = Client()

# prompt template for query rewriting
query_rewrite_template = """You are an AI assistant tasked with reformulating user queries to improve retrieval in a RAG system. 
Given the original query, rewrite it to be more specific, detailed, and likely to retrieve relevant information. 
Please only return the rewritten sentence.

Original query: {original_query}

Rewritten query:"""

query_rewrite_prompt = PromptTemplate(
    input_variables=["original_query"],
    template=query_rewrite_template
)

def rewrite_query(original_query):
    """
    Rewrite the original query to improve retrieval.
    
    Args:
    original_query (str): The original user query
    
    Returns:
    str: The rewritten query
    """
    prompt = query_rewrite_prompt.format(original_query=original_query)
    response = client.generate(model='llama3:latest', prompt=prompt)
    
    return response.response

# %%
original_query = "What are the impacts of climate change on the environment?"
rewritten_query = rewrite_query(original_query)
print("Original query:", original_query)
print("\nRewritten query:", rewritten_query)

# %% [markdown]
# ## Step-back Prompting
# 
# To generate broader, more general queries that can help retrieve relevant background information.

# %%
# prompt template for step-back prompting
step_back_template = """You are an AI assistant tasked with generating broader, more general queries to improve context retrieval in a RAG system.
Given the original query, generate a step-back query that is more general and can help retrieve relevant background information.
Please only return the step-back sentence.

Original query: {original_query}

Step-back query:"""

step_back_prompt = PromptTemplate(
    input_variables=["original_query"],
    template=step_back_template
)

def generate_step_back_query(original_query):
    """
    Rewrite the original query to improve retrieval.
    
    Args:
    original_query (str): The original user query
    
    Returns:
    str: The rewritten query
    """
    prompt = query_rewrite_prompt.format(original_query=original_query)
    response = client.generate(model='llama3:latest', prompt=prompt)
    
    return response.response

# %%
original_query = "What are the impacts of climate change on the environment?"
step_back_query = generate_step_back_query(original_query)
print("Original query:", original_query)
print("\nStep-back query:", step_back_query)

# %% [markdown]
# ## Sub-query Decomposition
# 
# To break down complex queries into simpler sub-queries for more comprehensive information retrieval.

# %%
# prompt template for sub-query decomposition
subquery_decomposition_template = """You are an AI assistant tasked with breaking down complex queries into simpler sub-queries for a RAG system.
Given the original query, decompose it into 2-4 simpler sub-queries that, when answered together, would provide a comprehensive response to the original query.
Please only return the decomposed sub-queries as the given example does.

Original query: {original_query}

example: What are the impacts of climate change on the environment?

Sub-queries:
1. What are the impacts of climate change on biodiversity?
2. How does climate change affect the oceans?
3. What are the effects of climate change on agriculture?
4. What are the impacts of climate change on human health?"""

subquery_decomposition_prompt = PromptTemplate(
    input_variables=["original_query"],
    template=subquery_decomposition_template
)

def decompose_query(original_query: str):
    """
    Decompose the original query into simpler sub-queries.
    
    Args:
    original_query (str): The original complex query
    
    Returns:
    List[str]: A list of simpler sub-queries
    """
    prompt = subquery_decomposition_prompt.format(original_query=original_query)
    response = client.generate(model='llama3:latest', prompt=prompt).response
    sub_queries = [q.strip() for q in response.split('\n') if q.strip() and not q.strip().startswith('Sub-queries:')]
    print(sub_queries)
    return sub_queries[1:]

# %%
original_query = "What are the impacts of climate change on the environment?"
sub_queries = decompose_query(original_query)
print("Original query:", original_query)
print("\nSub-queries:")
for i, sub_query in enumerate(sub_queries, 1):
    print(sub_query)


