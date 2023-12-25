from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser

from pydantic import BaseModel, Field
from typing import List


class HeritageAnswer(BaseModel):
    heritage_basis: List[str] = Field(description="Related heritages about the foreigner's case")
    heritage_solution: str = Field(description="Solution to client's case")
    conclusion: str = Field(description="Summarize key point in 1 sentence.")


def get_heritage_help_chain(llm: ChatOpenAI) -> LLMChain:
    parser = PydanticOutputParser(pydantic_object=HeritageAnswer)
    init_chat_template = """
    You are an AI guide who knows information about heritage of Seoul in Korea. 
    Your role is to give help foreigners who visit korea.

    Following dialog is summary of conversation where you and the foreigner.
    summary of conversation : ```
    {history}
    ```

    foreigner's inquiry related to heritage of Seoul : '{inquiry}'

    related heritage : ```
    {related_heritage}
    ```

    Considering context and knowledge basis, help your foreigner who doesn't know about Seoul heritage at all.
    And more, write your explain in markdown format to make it easy to understand.

    {format_instruction}
    """

    prompt = PromptTemplate(
        template=init_chat_template,
        input_variables=["inquiry", "related_heritage", "history"],
        partial_variables={"format_instruction": parser.get_format_instructions()},
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    chain = get_heritage_help_chain(ChatOpenAI())
    chain.run("What is the most famous heritage in Seoul?")
