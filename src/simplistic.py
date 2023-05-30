#%%
# SIMPLISTIC EXAMPLE
from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain, SimpleSequentialChain

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

print(catchphrase)
# %%
