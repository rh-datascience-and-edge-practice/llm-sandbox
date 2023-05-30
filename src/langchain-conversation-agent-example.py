# taken from here: https://python.langchain.com/en/latest/modules/agents/agents/examples/chat_conversation_agent.html

#%%
import os
from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from getpass import getpass

from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
SERPAPI_API_KEY = os.environ["SERPAPI_API_KEY"]

search = SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY)
tools = [
    Tool(
        name = "Current Search",
        func=search.run,
        description="useful for when you need to answer questions about current events or the current state of the world. the input to this should be a single search term."
    ),
]
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)

agent_chain = initialize_agent(tools, llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)

agent_chain.run(input="hi, i am bob")
# agent_chain.run(input="what's my name?")
# agent_chain.run("what are some good dinners to make this week, if i like thai food?")
# agent_chain.run(input="tell me the last letter in my name, and also tell me who won the world cup in 1978?")
# agent_chain.run(input="whats the weather like in pomfret?")
# %%
