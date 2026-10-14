import os
import uuid
import subprocess
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai.embeddings import OpenAIEmbeddings
from langgraph.types import interrupt, Command

load_dotenv()

llm = init_chat_model("openai:gpt-5-nano")

KNOWLEDGE = [
    # World War I
    "World War I began in 1914 after the assassination of Archduke Franz Ferdinand of Austria-Hungary.",
    "World War I lasted from 1914 to 1918.",
    "The assassination of Franz Ferdinand occurred in Sarajevo on June 28, 1914.",
    "The main Allied Powers in World War I included Britain, France, Russia, Italy, and later the United States.",
    "The main Central Powers included Germany, Austria-Hungary, the Ottoman Empire, and Bulgaria.",
    "The United States entered World War I in April 1917.",
    "Russia withdrew from World War I after the Bolshevik Revolution.",
    "The Treaty of Brest-Litovsk in 1918 ended the war between Russia and the Central Powers.",
    "Trench warfare was a defining feature of the Western Front.",
    "The Western Front stretched primarily through Belgium and northern France.",
    "The Battle of the Somme in 1916 was one of the largest and bloodiest battles of World War I.",
    "The Battle of Verdun in 1916 was one of the longest battles of World War I.",
    "Chemical weapons, including chlorine and mustard gas, were used extensively during World War I.",
    "Tanks were first used in combat by Britain during the Battle of the Somme in 1916.",
    "Military aircraft were initially used mainly for reconnaissance before becoming weapons of war.",
    "The sinking of the RMS Lusitania in 1915 contributed to tensions between Germany and the United States.",
    "The Zimmermann Telegram helped influence the United States toward entering World War I.",
    "An armistice ended the fighting on November 11, 1918.",
    "The Treaty of Versailles was signed in 1919 and formally ended the state of war between Germany and the Allied Powers.",
    "The Treaty of Versailles imposed territorial, military, and financial restrictions on Germany.",
    "The League of Nations was created following World War I.",
    "World War I caused the collapse of the German, Austro-Hungarian, Ottoman, and Russian Empires.",
    "The Spanish flu pandemic spread globally during and after World War I.",
    "Millions of military personnel and civilians died as a result of World War I.",
    # World War II
    "World War II lasted from 1939 to 1945.",
    "World War II began in Europe when Germany invaded Poland on September 1, 1939.",
    "Britain and France declared war on Germany on September 3, 1939.",
    "The major Axis Powers were Germany, Italy, and Japan.",
    "The major Allied Powers included Britain, France, the Soviet Union, the United States, and China.",
    "Germany invaded France and the Low Countries in 1940.",
    "The evacuation of Allied troops from Dunkirk occurred in 1940.",
    "The Battle of Britain was fought primarily between the British Royal Air Force and the German Luftwaffe in 1940.",
    "Operation Barbarossa was Germany's invasion of the Soviet Union, launched in June 1941.",
    "Japan attacked Pearl Harbor on December 7, 1941.",
    "The United States entered World War II after the attack on Pearl Harbor.",
    "The Battle of Stalingrad was fought between Germany and the Soviet Union from 1942 to 1943.",
    "The Battle of Midway in June 1942 was a major naval battle between the United States and Japan.",
    "The Allied invasion of Normandy, known as D-Day, took place on June 6, 1944.",
    "D-Day opened a major Western Allied front against Nazi Germany in occupied France.",
    "The Soviet Union advanced toward Germany from the Eastern Front.",
    "The Holocaust was the systematic persecution and murder of approximately six million Jews by Nazi Germany and its collaborators.",
    "Nazi Germany also persecuted and murdered millions of other people, including Roma, disabled people, Soviet prisoners of war, and political opponents.",
    "The Warsaw Ghetto Uprising took place in 1943.",
    "The Manhattan Project developed the first atomic bombs.",
    "The United States dropped an atomic bomb on Hiroshima on August 6, 1945.",
    "The United States dropped an atomic bomb on Nagasaki on August 9, 1945.",
    "Japan announced its surrender on August 15, 1945.",
    "Japan formally surrendered aboard the USS Missouri on September 2, 1945.",
    "Germany surrendered unconditionally in May 1945.",
    "World War II was the deadliest military conflict in history, with tens of millions of deaths.",
    "The United Nations was established in 1945 following World War II.",
    "The Nuremberg Trials prosecuted major Nazi leaders for war crimes and crimes against humanity.",
    "The war resulted in major changes to the political boundaries and balance of power in Europe and Asia.",
    "The Cold War between the United States and Soviet Union emerged in the aftermath of World War II.",
]

vector_store = InMemoryVectorStore(OpenAIEmbeddings(model="text-embedding-3-small"))
vector_store.add_documents(
    [Document(page_content=knowledge) for knowledge in KNOWLEDGE]
)


class IntentClassifier(BaseModel):
    message_intent: Literal["chat", "knowledge", "code"] = Field(
        ...,
        description="Classify whether the user wants to just chat, ask for knowledge or change code in the project.",
    )


class State(TypedDict):
    messages: Annotated[list, add_messages]
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


def accept_coding(state: State):
    user_prompt = state["messages"][-1].content
    decision = interrupt(
        f"About to run Claude Code with request:\n\n{user_prompt}\n\nDo you want to proceed? (yes/no)"
    )

    text = str(decision).strip().lower()

    if text in ["yes", "y", "proceed", "ok", "okay", "continue", "run"]:
        return {'next_node': 'code_agent'}
    elif text in ["no", "n", "cancel", "stop", "abort", "decline"]:
        return {
            "messages": [
                {"role": "assistant", "content": "Operation aborted by user."}
            ],
            "next_node": "denied",
        }
    else:
        return {"messages": [{"role": "user", "content": text}], "next_node": "accept_coding"}


def prompt_llm_chat(state: State):
    messages = [
        {"role": "system", "content": "You are a talktative chatbot for fun. Be nice."}
    ] + state["messages"]

    response = llm.invoke(messages)

    return {"messages": {"role": "assistant", "content": response.content}}


def prompt_llm_rag(state: State):
    query = state["messages"][-1].content
    documents = vector_store.similarity_search(query, k=3)

    context = "\n".join([doc.page_content for doc in documents])

    messages = [
        {
            "role": "system",
            "content": f"You're a RAG agent. Answer the user's question using only the context below, if the answer is not in the context, say 'I don't know'.\n\n{context}",
        }
    ] + state["messages"]

    response = llm.invoke(messages)

    return {"messages": {"role": "assistant", "content": response.content}}


def prompt_llm_code(state: State):
    user_prompt = state["messages"][-1].content
    workspace = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workspace")

    result = subprocess.run(
        ["claude", "-p", user_prompt, "--permission-mode", "acceptEdits"],
        cwd=workspace,
        capture_output=True,
        text=True,
    )

    output = result.stdout.strip() or result.stderr.strip()

    return {"messages": {"role": "assistant", "content": output}}


graph_builder = StateGraph(State)
graph_builder.add_node("classifier", classify_intent)
graph_builder.add_node("chat_agent", prompt_llm_chat)
graph_builder.add_node("rag_agent", prompt_llm_rag)
graph_builder.add_node("code_agent", prompt_llm_code)
graph_builder.add_node("accept_coding", accept_coding)

graph_builder.add_edge(START, "classifier")

graph_builder.add_conditional_edges(
    "accept_coding",
    lambda state: "end" if state.get("next_node") == "denied" else state["next_node"], {'end': END, 'code_agent': "code_agent","accept_coding": "accept_coding"},
)
graph_builder.add_conditional_edges(
    "classifier",
    lambda state: state["message_intent"],
    {"chat": "chat_agent", "knowledge": "rag_agent", "code": "accept_coding"},
)

graph_builder.add_edge("chat_agent", END)
graph_builder.add_edge("rag_agent", END)
graph_builder.add_edge("code_agent", END)

checkpointer = InMemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)

graph.get_graph().draw_mermaid_png(output_file_path="human_in_the_loop.png")

config = {"configurable": {"thread_id": str(uuid.uuid4())}}

while True:
    user_message = input("Enter your message: ")
    result = graph.invoke(
        {"messages": [{"role": "user", "content": user_message}]}, config=config
    )

    while "__interrupt__" in result:
        prompt = result["__interrupt__"][0].value
        decision = input(f"{prompt}\nYour decision: ")
        result = graph.invoke(Command(resume=decision), config=config)

    print(result["messages"][-1].content)
