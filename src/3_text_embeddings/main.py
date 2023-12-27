from langchain.document_loaders.text import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv


load_dotenv()

embedding_model = OpenAIEmbeddings()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=200, chunk_overlap=0)


loader = TextLoader(file_path="src/3_text_embeddings/facts.txt")
docs = loader.load_and_split(text_splitter=text_splitter)

for doc in docs:
    print(doc.page_content)
    print("\n")
