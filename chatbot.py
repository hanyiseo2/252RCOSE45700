import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "REMOVED_API_KEY"

print("🔄 벡터 스토어 로딩 중...")
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
print("✅ 벡터 스토어 로드 완료")

# LLM 설정
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.1
)

# Retriever 설정
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 7}  # 상위 7개 문서 검색
)

template = """You are an expert assistant for the AWS Well-Architected Framework.
Answer questions ONLY about AWS cloud architecture, best practices, and related technical topics.

**Critical Instructions:**
1. If the question is about AWS Well-Architected Framework, cloud architecture, or technical best practices:
   - Use ONLY information from the provided context
   - Provide detailed, structured answers with bullet points
   - Include specific AWS services when mentioned
   - Cite the source at the end

2. If the question is completely unrelated to AWS or cloud architecture (e.g., personal questions, general knowledge, non-technical topics):
   - Politely decline: "I'm specialized in AWS Well-Architected Framework topics. Please ask questions about cloud architecture, AWS best practices, security, cost optimization, reliability, performance, or operational excellence."

3. If the question is technical but context is insufficient:
   - State: "The provided AWS documents don't contain detailed information on this specific topic."

Context:
{context}

Question: {question}

Detailed Answer:"""

PROMPT = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# RAG Chain 생성
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

def ask_question(query):
    """질문을 받아 답변과 출처를 반환하는 함수"""
    print(f"\n❓ 질문: {query}")
    print("🔍 검색 중...\n")
    
    # RAG 실행
    result = qa_chain.invoke({"query": query})
    
    answer = result['result']
    source_docs = result['source_documents']
    
    # 답변 출력
    print(f"💡 답변:\n{answer}\n")
    
    # 출처 정리 및 출력
    print(f"📚 참조 출처:")
    
    sources = {}
    for doc in source_docs:
        source_url = doc.metadata.get('source', 'Unknown')
        source_name = source_url.split('/')[-1].replace('.html', '').replace('-', ' ').title()
        
        if source_url not in sources:
            sources[source_url] = {
                'name': source_name,
                'snippet': doc.page_content[:120].replace('\n', ' ')
            }
    
    for idx, (url, info) in enumerate(sources.items(), 1):
        print(f"\n  [{idx}] {info['name']}")
        print(f"      {url}")
        print(f"      \"{info['snippet']}...\"")
    
    return answer, list(sources.keys())

def main():
    """메인 실행 함수 - CLI 인터페이스"""
    print("\n" + "="*70)
    print("🤖 AWS Well-Architected Framework Chatbot")
    print("="*70)
    print("종료: 'quit', 'exit', 'q' 입력")
    print("\n💡 Example Questions:")
    print("  • What are security best practices for ML models?")
    print("  • How to optimize costs in generative AI workloads?")
    print("  • What is operational excellence in cloud architecture?")
    print("  • Compare traditional ML and generative AI security practices\n")
    
    while True:
        try:
            query = input("🧑 Your Question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q', '종료']:
                print("\n👋 Goodbye!")
                break
            
            if not query:
                print("⚠️  Please enter a question.\n")
                continue
            
            ask_question(query)
            print("\n" + "-"*70 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}\n")
            print("💡 Tip: Check if vectorstore exists and API key is valid\n")

if __name__ == "__main__":
    main()