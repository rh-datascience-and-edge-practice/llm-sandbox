#%%
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"]

#%%
# Proprietary LLM from e.g. OpenAI
# pip install openai
from langchain.llms import OpenAI
llm = OpenAI(model_name="text-davinci-003")

#%%
# Alternatively, open-source LLM hosted on Hugging Face
# pip install huggingface_hub
# from langchain import HuggingFaceHub
# llm = HuggingFaceHub(repo_id = "google/flan-t5-xl")

#%%
# The LLM takes a prompt as an input and outputs a completion
prompt = "Alice has a parrot. What animal is Alice's pet?"
completion = llm(prompt)
print(completion)

#%%
# Proprietary text embedding model from e.g. OpenAI
# pip install tiktoken

from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()
# The embeddings model takes a text as an input and outputs a list of floats
text = "Alice has a parrot. What animal is Alice's pet?"
text_embedding = embeddings.embed_query(text)
print(text_embedding)


# %%
# SIMPLISTIC EXAMPLE
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI

template = "What is a good name for a company that makes {product}?"
llm = OpenAI(model_name="text-davinci-003")
prompt = PromptTemplate(
    input_variables=["product"],
    template=template,
)

prompt.format(product="colorful socks")

chain = LLMChain(llm = llm, 
                  prompt = prompt)
# Create a second chain with a prompt template and an LLM
second_prompt = PromptTemplate(
    input_variables=["company_name"],
    template="Write a catchphrase for the following company: {company_name}",
)

chain_two = LLMChain(llm=llm, prompt=second_prompt)

# Combine the first and the second chain 
overall_chain = SimpleSequentialChain(chains=[chain, chain_two], verbose=True)

# Run the chain specifying only the input variable for the first chain.
catchphrase = overall_chain.run("colorful socks")
# Run the chain only specifying the input variable.
response = chain.run("colorful socks")

print(response)


# %%
# MORE COMPLEX EXAMPLE
from langchain.llms import OpenAI
llm = OpenAI(model_name="text-davinci-003")
from langchain import PromptTemplate, FewShotPromptTemplate

examples = [
    {"word": "happy", "antonym": "sad"},
    {"word": "tall", "antonym": "short"},
]

example_template = """
Word: {word}
Antonym: {antonym}\n
"""

example_prompt = PromptTemplate(
    input_variables=["word", "antonym"],
    template=example_template,
)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Give the antonym of every input",
    suffix="Word: {input}\nAntonym:",
    input_variables=["input"],
    example_separator="\n",
)

prompt = few_shot_prompt.format(input="capacious")
completion = llm(prompt)
print(completion)
# %%
