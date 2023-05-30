#%%
from langchain import ConversationChain
from langchain.llms import OpenAI
llm = OpenAI(model_name="text-davinci-003")

conversation = ConversationChain(llm=llm, verbose=True)

conversation.predict(input="Alice has a parrot.")

conversation.predict(input="Bob has two cats.")
conversation.predict(input="Brady has two dogs.")
conversation.predict(input="Trevor has one dog and one cat.")
conversation.predict(input="David has one cat.")
conversation.predict(input="John has two dogs.")

conversation.predict(input="How many pets total do these people have together?")