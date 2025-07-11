# Similar to Recursive character text splitting but with different set of separator.

from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="google/gemma-2-2b-it", task="text-generation")

text = """
class PDFTextProcessor:
    def __init__(self, pdf_path, model_repo="google/gemma-2-2b-it"):
        load_dotenv()
        self.pdf_path = pdf_path
        self.loader = PyPDFLoader(pdf_path)
        self.llm = HuggingFaceEndpoint(repo_id=model_repo, task="text-generation")
        self.documents = []
        self.splits = []

    def load_pdf(self):
        self.documents = self.loader.load()
        print(f"[INFO] Loaded {len(self.documents)} pages from '{self.pdf_path}'.")

if __name__ == "__main__":
    processor = PDFTextProcessor("2D Array Class summary - Audio.pdf")
    processor.load_pdf()
    processor.split_text()
    processor.query_model("Summarize this content in bullet points.")


"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=300, chunk_overlap=0
)
result = splitter.split_text(text)

print(len(result))

print(result[0])
