import uuid
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

llm = init_chat_model("openai:gpt-5-nano")


class IntentClassifier(BaseModel):
    message_intent: Literal["chat", "knowledge", "code"] = Field(
        ...,
        description="Classify whether the user wants to just chat, ask for knowledge or change code in the project.",
    )


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    message_intent: str | None


def classify_intent(state: State):
    structured_llm = llm.with_structured_output(IntentClassifier)
    result = structured_llm.invoke(
        [
            {
                "role": "user",
                "content": "Determine / classify wheter the user wants to chat ('chat'), ask for knowledge ('knowledge') or change code in the project ('code').",
            },
            {"role": "user", "content": state["messages"][-1].content},
        ]
    )

    return {"message_intent": IntentClassifier.model_validate(result).message_intent}


def prompt_llm_chat(state: State):
    messages = [
        {"role": "system", "content": "You are a talktative chatbot for fun. Be nice."}
    ] + state["messages"]

    response = llm.invoke(messages)

    return {"messages": {"role": "assistant", "content": response.content}}


def prompt_llm_rag(state: State):
    messages = [
        {
            "role": "system",
            "content": "No matter what the user says, always say 'I am the RAG agent'.",
        }
    ] + state["messages"]

    response = llm.invoke(messages)

    return {"messages": {"role": "assistant", "content": response.content}}


def prompt_llm_code(state: State):
    messages = [
        {
            "role": "system",
            "content": "No matter what the user says, always say 'I am the CODING agent'.",
        }
    ] + state["messages"]

    response = llm.invoke(messages)

    return {"messages": {"role": "assistant", "content": response.content}}


graph_builder = StateGraph(State)
graph_builder.add_node("classifier", classify_intent)
graph_builder.add_node("chat_agent", prompt_llm_chat)
graph_builder.add_node("rag_agent", prompt_llm_rag)
graph_builder.add_node("code_agent", prompt_llm_code)

graph_builder.add_edge(START, "classifier")
graph_builder.add_conditional_edges(
    "classifier",
    lambda state: state["message_intent"],
    {"chat": "chat_agent", "knowledge": "rag_agent", "code": "code_agent"},
)

graph_builder.add_edge("chat_agent", END)
graph_builder.add_edge("rag_agent", END)
graph_builder.add_edge("code_agent", END)

checkpointer = InMemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": str(uuid.uuid4())}}

while True:
    user_message = input("Enter your message: ")
    result = graph.invoke(
        {"messages": [{"role": "user", "content": user_message}]}, config=config
    )
    print(result["messages"][-1].content)
