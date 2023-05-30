#%%
from langchain.document_loaders import YoutubeLoader
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
embeddings = OpenAIEmbeddings()
llm = OpenAI(model_name="text-davinci-003")

loader = YoutubeLoader.from_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    
documents = loader.load()
# text_embedding = embeddings.embed_query(documents)
# create the vectorestore to use as the index
db = FAISS.from_documents(documents, embeddings)

retriever = db.as_retriever()

qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever, 
    return_source_documents=True)

query = "What am I never going to do?"
result = qa({"query": query})

print(result['result'])
# %%
