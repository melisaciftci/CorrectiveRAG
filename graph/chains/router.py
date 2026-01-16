from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

# Pydantic: bir değişkenin, classın vs tipini belirtip, bu tipte kullanılması zorunlu hale getiriyoruz.

class RouteQuery(BaseModel): # Dokümasyonu yazmaya alışmak zorundasın
    '''
    Route a user query to the most relevant datasource
    '''
    datasource : Literal["vectorstore", "websearch"] = Field( # Ya vectorstore olacak ya da websearch
        ..., # 3 nokta demek, burası vectorstore veya websearch olacak demek
        description = "Given a user question choose to route it to web search or a vectorstore"
    )

llm = ChatOpenAI(temperature=0)
structured_llm_router = llm.with_structured_output(RouteQuery) # llmin cevabını yapısal olarak ele alacağım ve ele alacağım sınıf da RouteQuery

system_prompt = """
You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to agents, prompt engineering and adversarial attacks on llms.
Use the vectorestore for questions on these topics. For all else, use web-search.
"""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human","{question}")
    ]
)

question_router = route_prompt | structured_llm_router

if __name__=="__main__":
    print(question_router.invoke(
        {"question":"what is agent memory"}
    ))