from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from dotenv import load_dotenv
from ingestion import retriever

load_dotenv()

llm = ChatOpenAI(temperature=0)

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieval documents"""

    binary_score : str = Field( # bool da olur str de
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

structured_llm_grader = llm.with_structured_output(GradeDocuments)

system_prompt = """
You are a grader assessing whether an LLM generation 
is grounded in / supported by a set of retrieved facts.
If the document contains keyword or semantics meaning related to question, grade it to relevant.
Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / 
supported by the set of facts.
"""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ('system', system_prompt),
        ('human', 'Retrieved document: {document} User question: {question}')
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader

if __name__=="__main__":
    user_question = "what is prompt engineering?"
    docs = retriever.get_relevant_documents(user_question)
    retrieved_document = docs[0].page_content
    print(retrieval_grader.invoke(
        {"question": user_question, "document": retrieved_document}
    ))