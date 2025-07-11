# Similar to Recursive character text splitting but with different set of separator.

from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="google/gemma-2-2b-it", task="text-generation")

text = """
### 🔍 What Is Similarity Search in Vector Databases?

**Similarity search** (also known as **vector similarity search**) is the process of finding the **most semantically similar** vectors (representing text, images, or other data) to a given query vector from a **vector database**.

In RAG and LLM-based applications, similarity search is used to **retrieve relevant context** from a knowledge base before generating a response.

---

## 📦 How It Works (Text-Based)

1. **Convert text chunks into embeddings** (vectors) using an embedding model.
2. **Store these vectors** in a vector database like FAISS, Chroma, or Pinecone.
3. **Embed the user query** using the same model.
4. **Compare the query vector** to all stored vectors to find the most similar ones.
5. **Retrieve the top-k** most relevant chunks to feed into the LLM.

---

## 📐 What Does “Similar” Mean?

Similarity is measured by **distance between vectors** in high-dimensional space:

| Metric                 | Meaning                                 | Used When                             |
| ---------------------- | --------------------------------------- | ------------------------------------- |
| **Cosine Similarity**  | Measures angle between vectors          | Most common in NLP (semantic meaning) |
| **Euclidean Distance** | Straight-line distance                  | Good when magnitude matters           |
| **Dot Product**        | Raw projection of one vector on another | Fast, used in dense retrieval         |

Example:

* `query = "Who is the CEO?"`
* Find chunks with cosine similarity ≥ 0.85 to query embedding.

---

## 📚 Example with FAISS in Python

```python
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# Embed documents and store
db = FAISS.from_documents(docs, OpenAIEmbeddings())

# Create a retriever
retriever = db.as_retriever(search_type="similarity", k=3)

# Perform similarity search
results = retriever.get_relevant_documents("What are the side effects?")
```

---

## 🛠 Popular Vector Databases

| Vector DB    | Notes                                                      |
| ------------ | ---------------------------------------------------------- |
| **FAISS**    | Facebook's library; local, fast, simple                    |
| **Chroma**   | Open-source, easy to use, LangChain-compatible             |
| **Pinecone** | Fully managed, scalable, fast, cloud-based                 |
| **Weaviate** | Open-source + cloud, includes filters and hybrid search    |
| **Qdrant**   | Rust-based, fast and filterable, supports payload metadata |
| **Milvus**   | Designed for enterprise-scale use                          |

---

## 🧠 Summary

**Similarity search** is the backbone of the **retrieval** part in **RAG** systems. It helps retrieve **relevant, semantically close** chunks based on a user query—without needing exact keyword matches.

---

Would you like:

* Code comparison of FAISS vs Pinecone?
* To combine keyword + vector search (hybrid)?
* An example with image or audio vectors?

Let me know how you're planning to use similarity search, and I’ll tailor the example.


"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN, chunk_size=300, chunk_overlap=0
)
result = splitter.split_text(text)

print(len(result))

print(result[0])
