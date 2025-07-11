

# 📚 LangChain Text Splitters 

In large language model (LLM) workflows, **text splitting** is critical when dealing with long documents. LangChain provides multiple **text splitter strategies** depending on the type and structure of your input — whether it's raw text, structured markdown, or code.

This folder contains examples of the following splitter types:

---

## 🧱 Available Text Splitters

| Splitter Type                    | Use Case                                 | Customization                                   |
| -------------------------------- | ---------------------------------------- | ----------------------------------------------- |
| `CharacterTextSplitter`          | Basic splitting based on character count | ✅ Chunk size, overlap, separator                |
| `RecursiveCharacterTextSplitter` | Smarter splits on structural tokens      | ✅ Language-aware, fallback strategies           |
| `SemanticChunker`                | Semantic similarity-based splitting      | ✅ Embedding-aware, more accurate context chunks |

---

## 📁 Files & What They Do

---

### 1. `length_based_splitter.py`

#### 🔍 What it does:

* Loads a PDF using `PyPDFLoader`
* Splits the content into chunks of **200 characters** using a **basic character-level splitter**

#### 📌 Split Strategy:

* Split by whitespace (`separator=" "`)
* No overlap between chunks (`chunk_overlap=0`)

#### ✨ Use Case:

Use when content is unstructured and you want quick, even-sized chunks.

#### ✅ Output:

```python
splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0, separator=" ")
result = splitter.split_documents(docs)
```

---

### 2. `text_structure_based.py`

#### 🔍 What it does:

* Similar to the above but uses **`RecursiveCharacterTextSplitter`** which tries to break at more meaningful points first (e.g., newlines, paragraphs).

#### 📌 Split Strategy:

* Attempts split at paragraphs, then sentences, then characters if needed

#### ✨ Use Case:

Works well for content like PDF reports where sections or paragraphs matter.

#### ✅ Output:

```python
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
result = splitter.split_documents(docs)
```

---

### 3. `markdown_splitting.py`

#### 🔍 What it does:

* Uses `RecursiveCharacterTextSplitter.from_language()` with `Language.MARKDOWN`
* Splits a blog-like markdown text into chunks

#### ✨ Use Case:

Works best for structured content like blogs, articles, or docs with headers and bullet points.

#### ✅ Output:

```python
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN, chunk_size=300, chunk_overlap=0
)
result = splitter.split_text(markdown_text)
```

---

### 4. `python_code_splitting.py`

#### 🔍 What it does:

* Splits Python code into semantically coherent blocks using the same recursive splitter, but with `Language.PYTHON`

#### ✨ Use Case:

Great for LLMs that need to analyze or explain Python code — avoids breaking mid-function or mid-class.

#### ✅ Output:

```python
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=300, chunk_overlap=0
)
result = splitter.split_text(python_code)
```

---

### 5. `semantic_meaning_based.py`

#### 🔍 What it does:

* Uses **`SemanticChunker`** from `langchain_experimental`
* Splits text using **embedding similarity**, keeping semantically related ideas together

#### 📌 Split Strategy:

* Based on `sentence-transformers/all-MiniLM-L6-v2`
* Uses statistical deviation of similarity to decide breakpoints

#### ✨ Use Case:

Best for nuanced content where chunking by structure/length could lose context (e.g., essay, newsletter, opinion).

#### ✅ Output:

```python
splitter = SemanticChunker(embedding_model)
result = splitter.split_text(text)
```

---

## 🧠 When to Use What?

| Use Case                                 | Best Splitter                                                   |
| ---------------------------------------- | --------------------------------------------------------------- |
| Basic fixed-length chunks                | `CharacterTextSplitter`                                         |
| Paragraphs, Markdown, logical splits     | `RecursiveCharacterTextSplitter`                                |
| Python/JS/Code files                     | `RecursiveCharacterTextSplitter.from_language(Language.PYTHON)` |
| Semantically meaningful paragraph chunks | `SemanticChunker`                                               |

---

## 🔚 Summary

Text splitters allow you to break documents into LLM-manageable units while preserving **coherence** and **meaning**. This step is **crucial for RAG pipelines**, summarization, and chunk-based retrieval.

In this repo:

* You've explored all major splitter types
* Used both syntactic and semantic approaches
* Applied them on PDFs, markdown docs, and code


--

