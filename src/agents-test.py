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
# creating an agent with a memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
agent = initialize_agent(tools, 
                         llm, 
                         agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, 
                         verbose=False,
                         memory=memory,
                         )
# NOTE: apparently LLMs sometimes don't follow instructions when generating formatted output.
# When the agent is using a tool, the responses from using the tool should be formatted in a specific way.
# We encountered some errors where the formatting instructions were NOT followed.
# After some investigation, it seems that this is a known issue with LLMs. There was a helpful comment in a Github issue.
# See:
# https://github.com/hwchase17/langchain/issues/1477#issuecomment-1459687475
# This method of "reinforcing" to the LLM what the output should look like seems to work.

# see this free course for additional free learning on "prompt engineering":
# https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/

agent.run(input="Who was the first president of the United States? Your response should include the prefix 'AI: <response>'.")
# agent.run(input="Who was the current?")
# %%
