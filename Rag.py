import os
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

class RAG:
    def __init__(self):
        self.embreddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

        self.vector_store = QdrantVectorStore.from_existing_collection(
            embedding=self.embreddings,
            collection_name=os.getenv("collection_name"),
            url=os.getenv("qdranturl"),
            api_key=os.getenv("qdrant_API_KEY"),
            prefer_grpc=True
        )

        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 10})

        self.llm = ChatOpenAI(
            model="lightning-ai/gpt-oss-120b",
            base_url="https://lightning.ai/api/v1",
            api_key=os.getenv("LLM_API_KEY"), 
            temperature=0.3
        )

        template = """
        คุณคือ "ชายลึกลับ" ชายที่คอยช่วยเหลือผู้กล้า ในเกม Headliner
        หน้าที่ของคุณคือช่วยผู้เล่นตอบคำถามเรื่องความรู้ทางการเงินของประเทศไทย โดยอ้างอิงจากข้อมูลที่ได้รับเท่านั้นข้อมูลอ้างอิง (Context):{context}
        คำถามของผู้เล่น: {question}
        ข้อกำหนดการตอบ:
            1. ตอบให้ถูกต้องตามกฎหมายและตัวเลขใน Context เป๊ะๆ
            2. ใช้ภาษาพูดที่น่ารัก เป็นกันเอง (ลงท้ายด้วย ครับ ตามความเหมาะสม)
            3. ถ้าข้อมูลใน Context ไม่พอ ให้ตอบว่า "เรื่องนี้ฉันยังไม่มีข้อมูลในคัมภีร์ ลองถามเรื่องอื่นดูนะ" (ห้ามมั่วเองเด็ดขาด) หรือ คิดคำถามแนะนำให่ผู้เล่นเพื่อคำถามที่ดีกว่า เพื่อให้ผู้เล่นได้คำตอบ หรือคิดคำถามเพื่อแนะนำผุ้เล่นเพื่อให้นายตอบผุ้เล่นได้
            4. ถ้ามีการคำนวณ ไม่คำนวณให้ อยากให้บอกมากกว่าว่าคำนวณยังไง
            5. **อนุญาตให้เทียบเคียงตารางภาษีให้ผู้เล่นได้** (เช่น ถ้าผู้เล่นบอกเงินเดือน ให้ลองเทียบกับตารางเงินได้สุทธิบ หรืออัตราการเสียภาษีเงินได้บุคคลธรรมดา เพื่อประมาณการ)
            6. ถ้ามีการกดเข้่า link ต่อไป ห้ามตอบให้ผู้เล่นเด็ดขาด**

        คำตอบ:
        """

        prompt = ChatPromptTemplate.from_template(template)

        self.chain = (
            {"context": self.retriever | self._format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def _format_docs(self, docs):
        return "\n\n".join([d.page_content for d in docs])
    
    def ask(self,question: str):
        return self.chain.invoke(question)

bot = RAG()