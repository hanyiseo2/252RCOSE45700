import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY 환경변수를 설정해주세요!")

print("📥 데이터 로딩 시작...")

local_pdfs = [
    "./docs/wellarchitected-machine-learning-lens.pdf",
    "./docs/generative-ai-lens.pdf",
    "./docs/responsible-ai-lens.pdf",
]

urls = [
    # === Lenses ===
    "https://docs.aws.amazon.com/wellarchitected/latest/responsible-ai-lens/responsible-ai-lens.html",
    "https://docs.aws.amazon.com/wellarchitected/latest/generative-ai-lens/generative-ai-lens.html",
    "https://docs.aws.amazon.com/wellarchitected/latest/machine-learning-lens/machine-learning-lens.html",
    "https://docs.aws.amazon.com/wellarchitected/latest/modern-industrial-data-technology-lens/modern-industrial-data-technology-lens.html",
    "https://docs.aws.amazon.com/wellarchitected/latest/end-user-computing-lens/end-user-computing-lens.html",
    "https://docs.aws.amazon.com/wellarchitected/latest/supply-chain-lens/supply-chain-lens.html",
    "https://docs.aws.amazon.com/wellarchitected/latest/iot-lens/iot-lens.html",
    "https://docs.aws.amazon.com/wellarchitected/latest/high-performance-computing-lens/high-performance-computing-lens.html?did=wp_card&trk=wp_card",
    "https://docs.aws.amazon.com/wellarchitected/latest/mergers-and-acquisitions-lens/mergers-and-acquisitions-lens.html?did=wp_card&trk=wp_card",
    "https://docs.aws.amazon.com/wellarchitected/latest/migration-lens/migration-lens.html?did=wp_card&trk=wp_card",
    "https://docs.aws.amazon.com/wellarchitected/latest/government-lens/government-lens.html?did=wp_card&trk=wp_card",
    "https://docs.aws.amazon.com/wellarchitected/latest/connected-mobility-lens/connected-mobility-lens.html?did=wp_card&trk=wp_card",
    "https://docs.aws.amazon.com/wellarchitected/latest/analytics-lens/analytics-lens.html?did=wp_card&trk=wp_card",
    ]

all_docs = []

if local_pdfs:
    print("📄 로컬 PDF 로딩 중...")
    for pdf_path in local_pdfs:
        if os.path.exists(pdf_path):
            try:
                loader = PyPDFLoader(pdf_path)
                docs = loader.load()
                all_docs.extend(docs)
                print(f"  ✅ {pdf_path}: {len(docs)}페이지")
            except Exception as e:
                print(f"  ❌ {pdf_path}: {e}")
        else:
            print(f"  ⚠️  {pdf_path} 파일을 찾을 수 없습니다")

# 웹페이지 로드
successful_docs = []
failed_urls = []

print(f"📄 총 {len(urls)}개 URL 로딩 시작...\n")

for idx, url in enumerate(urls, 1):
    try:
        url_name = url.split('/')[-1][:50]
        print(f"  [{idx}/{len(urls)}] {url_name}...", end=" ")
        loader = WebBaseLoader([url])
        docs = loader.load()
        all_docs.extend(docs)
        print("✅")
    except Exception as e:
        print(f"❌ {str(e)[:30]}")
        failed_urls.append(url)

print(f"\n✅ {len(all_docs)}개 문서 로드 완료")
if failed_urls:
    print(f"⚠️  {len(failed_urls)}개 URL 로드 실패:")
    for url in failed_urls:
        print(f"   - {url.split('/')[-1]}")

# 텍스트 청킹
print("\n✂️ 문서 청킹 중...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = text_splitter.split_documents(all_docs)
print(f"✅ {len(chunks)}개 청크 생성 완료")

# 청크 분포 확인
print("\n📋 청크 분포 (상위 10개):")
source_count = {}
for chunk in chunks:
    source = chunk.metadata.get('source', 'Unknown')
    source_name = source.split('/')[-1].replace('.html', '').replace('-', ' ').title()
    source_count[source_name] = source_count.get(source_name, 0) + 1

for idx, (name, count) in enumerate(sorted(source_count.items(), key=lambda x: x[1], reverse=True)[:10], 1):
    print(f"  {idx}. {name}: {count}개")

if len(source_count) > 10:
    print(f"  ... 외 {len(source_count) - 10}개")

# 임베딩 생성
print("\n🔢 임베딩 생성 중... (약 30초~1분 소요)")
embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )
vectorstore = FAISS.from_documents(chunks, embeddings)
print("✅ 벡터 스토어 생성 완료")

# 저장
print("\n💾 벡터 스토어 저장 중...")
vectorstore.save_local("vectorstore")
print("✅ 저장 완료: ./vectorstore/")

# 통계
avg_chunk_size = sum(len(c.page_content) for c in chunks) // len(chunks)
print("\n" + "="*60)
print("🎉 모든 작업 완료!")
print(f"  📄 문서: {len(all_docs)}개")
print(f"  ✂️  청크: {len(chunks)}개")
print(f"  📏 평균 청크 크기: {avg_chunk_size}자")
print(f"  📂 저장 위치: ./vectorstore/")
print("="*60)