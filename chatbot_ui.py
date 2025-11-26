import gradio as gr
from chatbot import qa_chain

print("🚀 Gradio UI 시작 중...\n")

def chat_interface(message, history):
    """Gradio 채팅 인터페이스 - chatbot.py의 qa_chain 재사용"""
    try:
        # chatbot.py에서 이미 설정된 qa_chain 사용
        result = qa_chain.invoke({"query": message})
        answer = result['result']
        source_docs = result['source_documents']
        
        # 출처 정리
        sources = set()
        for doc in source_docs:
            source_name = doc.metadata.get('source', 'Unknown').split('/')[-1].replace('.html', '').replace('-', ' ').title()
            sources.add(source_name)
        
        # 출처 포맷팅
        source_text = "\n".join([f"• {s}" for s in sorted(sources)[:5]])
        
        # 최종 응답
        response = f"{answer}\n\n---\n📚 **Sources:**\n{source_text}"
        
        return response
        
    except Exception as e:
        return f"❌ Error: {str(e)}\n\nPlease check:\n1. Vectorstore exists\n2. API key is valid"

# Gradio 인터페이스 (버전 호환)
demo = gr.ChatInterface(
    fn=chat_interface,
    title="🤖 AWS Well-Architected Framework Chatbot",
    description="Ask questions about AWS Well-Architected best practices, pillars, and lenses.",
    examples=[
        "What are the core principles of Responsible AI?",
        "How to implement security best practices for ML models?",
        "What are cost optimization strategies for generative AI?",
        "Compare operational excellence between traditional ML and GenAI",
        "What is the shared responsibility model in AWS?",
    ],
)

if __name__ == "__main__":
    print("✅ Gradio UI 준비 완료")
    print("📍 Local access: http://127.0.0.1:7860")
    print("📍 Public access: http://13.221.65.74:7860\n")
    demo.launch(
        share=False,
        server_name="0.0.0.0",  # 모든 네트워크 인터페이스에서 접속 허용
        server_port=7860
    )