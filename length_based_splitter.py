from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="google/gemma-2-2b-it", task="text-generation")

loader = PyPDFLoader("textpdf.pdf")
docs = loader.load()

splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0, separator=" ")
result = splitter.split_documents(docs)

print(result[0])
print(result[0].page_content)
print(result)
