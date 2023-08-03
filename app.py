# app.py
import langchain
langchain.verbose = False
from typing import List, Union
import os
from dotenv import load_dotenv, find_dotenv
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
from langchain import LLMChain
from langchain.llms import LlamaCpp,ExLLamaLLM
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant
import qdrant_client
from langchain.chains import RetrievalQA
from langchain import PromptTemplate




callbacks = [StreamingStdOutCallbackHandler()]

def init_page() -> None:
    st.set_page_config(
        page_title="Llama V2"
    )
    st.header("Llama V2")


def init_messages() -> None:
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(
                content="You are a helpful assistant. Please answer the question precisely and concisely.")
        ]



def get_answer(llm, messages, documents) -> str:
    query = llama_v2_prompt(convert_langchainschema_to_dict(messages), documents)
    prompt_template = "{dummy}"
    prompt = PromptTemplate(
    input_variables=["dummy"], template=prompt_template
    )
    llmchain = LLMChain(llm=llm, prompt=prompt)
    #qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    answer = llmchain.run(query)
    #res = qa(messages) 
    document = documents[0].metadata["source"]
    answer += "\n\nDocument referenced: "+document


    return answer


def find_role(message: Union[SystemMessage, HumanMessage, AIMessage]) -> str:
    """
    Identify role name from langchain.schema object.
    """
    if isinstance(message, SystemMessage):
        return "system"
    if isinstance(message, HumanMessage):
        return "user"
    if isinstance(message, AIMessage):
        return "assistant"
    raise TypeError("Unknown message type.")


def convert_langchainschema_to_dict(
        messages: List[Union[SystemMessage, HumanMessage, AIMessage]]) \
        -> List[dict]:
    """
    Convert the chain of chat messages in list of langchain.schema format to
    list of dictionary format.
    """
    return [{"role": find_role(message),
             "content": message.content
             } for message in messages]


def llama_v2_prompt(messages: List[dict], documents: List) -> str:
    """
    Convert the messages in list of dictionary format to Llama2 compliant format.
    """
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    BOS, EOS = "<s>", "</s>"
    DEFAULT_SYSTEM_PROMPT = f"""You are a helpful, respectful and honest assistant of the company Decision Inc that would answer questions regarding the company in a chat manner. Always answer as helpfully as possible, while being safe. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. Please simply answer the questions without reviewing your response or additional context."""
    context = """Please answer the question using these contents that were queried for you. You may ignore them if the contents were irrelevant and answer on your own: \n"""
    for doc in documents:
        context += doc.page_content + "\n"
    if messages[0]["role"] != "system":
        messages = [
            {
                "role": "system",
                "content": DEFAULT_SYSTEM_PROMPT,
            }
        ] + messages
    messages = [
        {
            "role": messages[1]["role"],
            "content": B_SYS + messages[0]["content"] + context+ E_SYS + messages[1]["content"],
        }
    ] + messages[2:]
    if len(messages) <= 6:
        messages_list = [
        f"{BOS}{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} {EOS}"
        for prompt, answer in zip(messages[::2], messages[1::2])
    ]
    else:
        messages_list = [
        f"{BOS}{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} {EOS}"
        for prompt, answer in zip(messages[-6::2], messages[-5::2])
    ]
    messages_list.append(
        f"{BOS}{B_INST} {(messages[-1]['content']).strip()} {E_INST}")

    return "".join(messages_list)


def main() -> None:
    _ = load_dotenv(find_dotenv())
    model_type = os.environ.get('MODEL_TYPE')
    model_path = os.environ.get('MODEL_PATH')
    model_n_ctx = os.environ.get('MODEL_N_CTX')
    n_gpu_layers = os.environ.get('MODEL_N_GPU')
    n_threads = os.environ.get('MODEL_THREAD')
    n_batch = os.environ.get('MODEL_N_BATCH')
    model_dir = os.environ.get('MODEL_DIR')
    temperature=os.environ.get("temperature")
    top_p=os.environ.get("top_p")
    top_k=os.environ.get("top_k")    

    init_page()
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    match model_type:
        case "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, 
                           n_ctx=model_n_ctx, 
                           n_gpu_layers=n_gpu_layers,
                           callback_manager=callback_manager,
                           verbose=False, 
                           n_threads=n_threads,
                           n_batch=n_batch, 
                           use_mmap=True,
                           temperature=temperature,
                           top_p=top_p,
                           top_k=top_k,)
        case "ExLlama":
            llm = ExLLamaLLM(
                model_path=model_path,
                model_dir=model_dir,
                stream_output=False,
                gpu_mem_allocation='1',
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                callback_manager=callback_manager
                
            )
    
    init_messages()

    # Supervise user input
    if user_input := st.chat_input("Input your question!"):
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner("Llama is typing ..."):
            documents = retriever.get_relevant_documents(user_input)
            answer = get_answer(llm, 
                                st.session_state.messages,
                                documents
                                #user_input
                                )
        st.session_state.messages.append(AIMessage(content=answer))

    # Display chat history
    messages = st.session_state.get("messages", [])

    for message in messages:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)




# streamlit run app.py
if __name__ == "__main__":
    _ = load_dotenv(find_dotenv())
    embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    client = qdrant_client.QdrantClient(url="localhost", port=6333
                                         #path='C:/Users/user/KnowledgeBot/db'
                                         )
    db = Qdrant(client=client, collection_name="myDocument", embeddings=embeddings)
    retriever = db.as_retriever()
    main()

