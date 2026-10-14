import uuid 
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

llm = init_chat_model("openai:gpt-5-nano")


def prompt_llm(state: MessagesState) -> str:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


graph_builder = StateGraph(MessagesState)
graph_builder.add_node(prompt_llm)
graph_builder.add_edge(START, "prompt_llm")
graph_builder.add_edge("prompt_llm", END)

checkpointer = InMemorySaver()

graph = graph_builder.compile(checkpointer=checkpointer)

config = {'configurable': {'thread_id': uuid.uuid4()}}

while True:
    user_message = input("Enter your message: ")
    print(graph.invoke({"messages": [{"role": "user", "content": user_message}]}, config=config))
    
