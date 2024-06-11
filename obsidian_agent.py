import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import DirectoryLoader

from langchain.vectorstores import FAISS
from langchain.retrievers.multi_query import MultiQueryRetriever  
from langchain_community.embeddings import GPT4AllEmbeddings, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

# 1. Vectorise the sales response csv data
loader = DirectoryLoader('/Users/rtadewald/Library/Mobile Documents/iCloud~md~obsidian/Documents/Zettelkasten', glob="**/*.md")
docs = loader.load()
len(docs)

embeddings = OpenAIEmbeddings()
# embeddings = GPT4AllEmbeddings()
db = FAISS.from_documents(docs, embeddings)
retriever = db.as_retriever()


def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )



llm = ChatOpenAI(temperature=0)  
retriever_from_llm = MultiQueryRetriever.from_llm(  
    retriever=retriever, llm=llm  
)

import logging  
logging.basicConfig()  
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
question = "Eu possuo algum documento que fale sobre Copywriting?"  

n_docs = retriever.invoke(question)
pretty_print_docs(n_docs)


from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import OpenAI

llm = OpenAI(temperature=0)
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

compressed_docs = compression_retriever.invoke(
    "É possível utilizar Machine Learning para melhorar no Trading?")

pretty_print_docs(compressed_docs)



# 3. Setup LLMChain & prompts
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

template = """
You are a world class business development representative. 
I will share a prospect's message with you and you will give me the best answer that 
I should send to this prospect based on past best practies, 
and you will follow ALL of the rules below:

1/ Response should be very similar or even identical to the past best practies, 
in terms of length, ton of voice, logical arguments and other details

2/ If the best practice are irrelevant, then try to mimic the style of the best practice to prospect's message

Below is a message I received from the prospect:
{message}

Here is a list of best practies of how we normally respond to prospect in similar scenarios:
{best_practice}

Please write the best response that I should send to this prospect:
"""

prompt = PromptTemplate(
    input_variables=["message", "best_practice"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)


# 4. Retrieval augmented generation
def generate_response(message):
    best_practice = retrieve_info(message)
    response = chain.run(message=message, best_practice=best_practice)
    return response


# 5. Build an app with streamlit
def main():
    st.set_page_config(
        page_title="Customer response generator", page_icon=":bird:")

    st.header("Customer response generator :bird:")
    message = st.text_area("customer message")

    if message:
        st.write("Generating best practice message...")

        result = generate_response(message)

        st.info(result)


if __name__ == '__main__':
    main()