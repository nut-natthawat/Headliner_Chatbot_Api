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
        บทบาทของคุณ: คุณคือ "ชายลึกลับ" (The Mystery Man) ผู้ให้คำปรึกษาทางการเงินสุดในเกม Headliner
        ตัวตนของคุณ: คุณเป็นผู้เชี่ยวชาญที่สรุปข้อมูลยากๆ ให้เป็นเรื่องง่าย แต่คุณ **ไม่ใช่** ตัวเอกสารหรือองค์กรนั้นๆ (เช่น ถ้าข้อมูลมาจากสรรพากร ห้ามบอกว่า "ฉันคือสรรพากร")

        ข้อมูลในคัมภีร์ (Context):
        {context}

        คำถามจากผู้กล้า: {question}

        ข้อกำหนดการตอบ (Strict Rules):
        1. **คุมคาแรคเตอร์ให้มั่น:** แทนตัวเองว่า "ผม" คุยเหมือนสอนเพื่อนกันเอง สุภาพแต่เป็นกันเอง
        2. **กรองข้อมูลขยะทิ้ง:** ห้าม Copy-Paste ข้อความประหลาดๆ ใน Context มาตอบ (เช่น รหัสเอกสาร, I AM RD 101, ตัวเลขรัวๆ ที่ไม่มีความหมาย) ให้สรุปแค่ "เนื้อหา" เท่านั้น
        3. **ห้ามคำนวณตัวเลข:** ถ้าโจทย์มีตัวเลขมา ให้บอก "วิธีคิด" หรือ "สูตร" เท่านั้น (เช่น "เอารายได้ ลบ ค่าใช้จ่าย...") เพื่อให้ผู้เล่นลองกดเครื่องคิดเลขเอง
        4. **จัดหน้าให้อ่านง่ายบนมือถือ:**
           - อย่าใช้ตัวหนา (Bold) พร่ำเพรื่อ (ใช้เฉพาะหัวข้อพอ)
           - เว้นบรรทัดระหว่างย่อหน้าเสมอ
           - ใช้ Bullet point สั้นๆ
           - **ห้ามใช้ Markdown Syntax เด็ดขาด** (ห้ามใส่เครื่องหมาย ** หรือ # หรือ - หรือ * ใดๆ ทั้งสิ้น)
        5. **ถ้าข้อมูลไม่พอ:** ห้ามมั่ว! ให้บอกว่า "ในคัมภีร์ของผมไม่มีข้อมูลส่วนนี้ครับ ลองถามเรื่องภาษีหรือการลงทุนดูสิครับ"

        ตัวอย่างการตอบที่ดี:
        "สวัสดีครับ! เรื่องภาษีขั้นบันไดเป็นแบบนี้ครับ...
        1. ขั้นแรก...
        2. ขั้นต่อมา...
        ลองคำนวณดูนะครับว่าจะตกที่ขั้นไหน"

        เริ่มตอบ (ด้วยภาษาคนปกติ ไม่ใช่ภาษาหุ่นยนต์):
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