"""
ingest.py — 학사일정 PDF를 Chroma 벡터스토어에 적재 (OpenAI Embedding 사용)
================================================
실행 방법:
  python ingest.py

주의: 임베딩 모델을 바꾸면 벡터 공간이 달라지므로,
기존에 다른 임베딩으로 만든 ./chroma_db 폴더가 있다면 삭제 후 다시 실행하세요.
"""

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

PDF_PATH = "schedule.pdf"  # 학사일정 PDF 경로

loader = PyPDFLoader(PDF_PATH)
documents = loader.load()
print("페이지 수 :", len(documents))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""],
)

split_docs = splitter.split_documents(documents)
print("분할 문서 :", len(split_docs))

# OpenAI 임베딩
embedding = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vectorstore = Chroma.from_documents(
    documents=split_docs,
    embedding=embedding,
    persist_directory="./chroma_db",
    collection_name="pdf_collection",
)

print("Chroma 저장 완료 (collection: pdf_collection)")