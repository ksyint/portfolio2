# help from https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps
# help from https://github.com/langchain-ai/streamlit-agent/blob/main/streamlit_agent/basic_memory.py
from jinja2 import Template
import json
import os

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import (
    ConversationBufferMemory,
    ConversationSummaryMemory,
    ChatMessageHistory,
)
from langchain.schema.document import Document

from typing import Any, Dict, List
from dotenv import load_dotenv

from custom_chains.retriever import (
    get_chroma_retriever,
    get_pinecone_retriever,
    get_faiss_ensemble_retriever,
)
from custom_chains.chat import get_heritage_help_chain


load_dotenv()


class HeritageChatbot:
    def __init__(self, model="gpt-4", verbose: bool = False):
        self.model = model
        self.verbose = verbose
        self.sum_memory = ConversationSummaryMemory(
            llm=ChatOpenAI(temperature=0, model=self.model),
            human_prefix="foreigner",
            ai_prefix="Guide",
        )
        # self.prev_summary = "Dialog is not started yet."

        self.llm_advisor = ChatOpenAI(temperature=0, model=self.model)
        self.embedding_model = OpenAIEmbeddings()

        # self.retriever = get_faiss_ensemble_retriever(self.embedding_model)
        # self.retriever = get_pinecone_retriever(self.embedding_model)
        self.retriever = get_chroma_retriever(self.embedding_model)
        self.heritage_help_chain = get_heritage_help_chain(self.llm_advisor)


    def __proc_heritage_doc(self, doc:Document):
        src, _ = os.path.splitext(os.path.basename(doc.metadata["source"]))
        return "#"+src+'\n'+doc.page_content

    def __call__(self, inquiry: str) -> Dict:
        heritage_basis:List[Document] = self.retriever.get_relevant_documents(inquiry)
        heritage_basis:List[str] = list(map(self.__proc_heritage_doc, heritage_basis))
        heritage_basis = "\n\n\n".join(heritage_basis)

        eng_guide = self.heritage_help_chain.run(
            inquiry=inquiry,
            related_heritage=heritage_basis[:10],
            history=self.sum_memory.buffer
        )
        eng_guide_dict = json.loads(eng_guide)
        eng_guide_format = """
                            ### Brief Explanation
                            {{ heritage_explanation }}
                            
                            ### Supplement Information
                            {% for heritage_lists in related_heritage %}
                            - {{heritage_lists}}
                            {% endfor %}                            
                            """
        ## Explanation
        # { %
        # for guide in related_heritage %}
        # - {{heritage_explanation}}
        # { % endfor %}
        answer_template = Template(eng_guide_format)
        rendered_answer = answer_template.render(
            # conclusion=eng_guide_dict["conclusion"],
            heritage_explanation=eng_guide_dict["heritage_solution"],
            related_heritage=eng_guide_dict["heritage_basis"],
            # related_heritage=eng_guide_dict["heritage_basis"],
        )

        # 대화 내용 요약 후 메모리에 저장
        self.sum_memory.save_context(
            {"foreigner": inquiry}, {"guide": eng_guide_dict["conclusion"]}
        )
        self.sum_memory.chat_memory.messages[-1].additional_kwargs=eng_guide_dict
        self.sum_memory.predict_new_summary(
            messages=self.sum_memory.chat_memory.messages,
            existing_summary=self.sum_memory.buffer,
        )

        if self.verbose:
            print("#" * 100)
            print("heritage basis")
            print(heritage_basis)
            print()
            print("memory")
            for msg in self.sum_memory.chat_memory.messages:
                print(msg)
            print()
            print("buffer")
            print(self.sum_memory.buffer)
            print("#" * 100)

        return rendered_answer

    def __del__(self):
        print(self.sum_memory.buffer)
        print(self.sum_memory.chat_memory)
        print("Chatbot deleted properly")

if __name__ == "__main__":
    chatbot = HeritageChatbot(verbose=True)
    while True:
        query = input("ask something about heritage of Seoul : ")
        if query == "q":
            break
        answer = chatbot(query)
        print(answer)
