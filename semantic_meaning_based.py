from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import os

# Load .env variables
load_dotenv()

# Define the LLM (text generation model)
llm = HuggingFaceEndpoint(repo_id="google/gemma-2-2b-it", task="text-generation")

# Define the embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Define the semantic chunker
splitter = SemanticChunker(
    embedding_model,
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1,
)

# Sample text
text = """
Renewable energy and electric vehicles (EVs) are central to the global push for sustainability, though they address different sectors. Renewable energy sources like solar and wind aim to replace fossil fuels in electricity generation, reducing carbon emissions and dependency on non-renewable resources. Ancient civilizations, such as the Mesopotamians and the Egyptians, developed intricate writing systems that were crucial for communication, trade, and record-keeping. Cuneiform tablets and hieroglyphics not only offer insights into their cultures but also laid the foundation for modern language systems. These early innovations reflect humanity’s deep need to preserve knowledge and maintain societal order.

In recent years, the popularity of indoor plants has surged, especially among urban dwellers. Aside from their aesthetic appeal, studies have shown that houseplants can improve air quality, reduce stress, and boost productivity. The trend reflects a growing desire to reconnect with nature, even in compact living spaces like apartments and studios.
"""

# Run semantic splitting
result = splitter.split_text(text)

print(len(result))
print(result[1])
