from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Qdrant
from langchain.llms import GPT4All, LlamaCpp, ExLLamaLLM
import os
from langchain.embeddings import LlamaCppEmbeddings
import qdrant_client
load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

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

def main():
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    # embeddings = LlamaCppEmbeddings(model_path=model_path, n_ctx=model_n_ctx, n_gpu_layers=n_gpu_layers, n_threads=n_threads)
    client = qdrant_client.QdrantClient(url="localhost", port=6333
                                         #path='C:/Users/user/KnowledgeBot/db'
                                         )
    db = Qdrant(client=client, collection_name="myDocument", embeddings=embeddings)
    retriever = db.as_retriever()
    # Prepare the LLM
    callbacks = [StreamingStdOutCallbackHandler()]
    match model_type:
        case "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, n_gpu_layers=n_gpu_layers,callbacks=callbacks, verbose=False, n_threads=n_threads,n_batch=n_batch, use_mmap=True)
        case "ExLlama":
            llm = ExLLamaLLM(
                model_path=model_path,
                model_dir=model_dir,
                stream_output=False,
                gpu_mem_allocation='1',
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                callbacks=callbacks
                
            )
        case _default:
            print(f"Model {model_type} not supported!")
            exit;
    
    
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        
        # Get the answer from the chain
        res = qa(query)    
        answer, docs = res['result'], res['source_documents']

        # Print the result
        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        print(answer)
        
        # Print the relevant sources used for the answer
        for document in docs:
            print("\n> " + document.metadata["source"] + ":")
            print(document.page_content)

if __name__ == "__main__":
    main()
