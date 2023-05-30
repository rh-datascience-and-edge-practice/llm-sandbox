#%%
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
from langchain import ConversationChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
load_dotenv()

llm = OpenAI(model_name="text-davinci-003")

tools = load_tools(["wikipedia", "llm-math"], llm=llm)
conversation = ConversationChain(llm=llm, verbose=True)
# TODO: how do we give this agent memory?
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
agent = initialize_agent(tools, 
                         llm, 
                         agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, 
                         verbose=False,
                         memory=memory,
                         )

agent.run(input="Who was the first president of the United States?")
agent.run(input="Who was the second?")