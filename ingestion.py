import os
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# OpenAI API 키 확인
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY 환경변수를 설정해주세요!")

print("📥 데이터 로딩 시작...")

urls = [
    "https://docs.aws.amazon.com/wellarchitected/latest/responsible-ai-lens/responsible-ai-lens.html",
    "https://docs.aws.amazon.com/wellarchitected/latest/generative-ai-lens/generative-ai-lens.html",
    "https://docs.aws.amazon.com/wellarchitected/latest/machine-learning-lens/machine-learning-lens.html"
]

# 웹페이지 로드
loader = WebBaseLoader(urls)
documents = loader.load()
print(f"✅ {len(documents)}개 문서 로드 완료")

# 텍스트 청킹
print("✂️ 문서 청킹 중...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)
print(f"✅ {len(chunks)}개 청크 생성 완료")

# 임베딩 생성 및 벡터 스토어 구축
print("🔢 임베딩 생성 중... (시간이 걸릴 수 있습니다)")
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)
print("✅ 벡터 스토어 생성 완료")

# 로컬에 저장
print("💾 벡터 스토어 저장 중...")
vectorstore.save_local("vectorstore")
print("✅ 저장 완료: ./vectorstore/")

print("\n🎉 모든 작업 완료!")
print(f"총 {len(documents)}개 문서, {len(chunks)}개 청크 처리됨")