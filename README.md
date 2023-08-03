# Summary:
This was forked and modified from the famous PrivateGPT (https://github.com/imartinez/privateGPT) repo. 

The app is designed to be used specifically for Llama2 and in particular Exllama running GPTQ model at the moment.

Major changes have been made to ensure it works with a different and more performant vector database (Qdrant), different model and also stiched it a UI modified from this post (https://medium.com/@daydreamersjp/implementing-locally-hosted-llama2-chat-ui-using-streamlit-53b181651b4e)

The Llama2 specific prompt was modified from the one shared by mr96 (https://huggingface.co/mr96) in a post under here (https://huggingface.co/TheBloke/Llama-2-13B-chat-GPTQ/discussions/5) to incorporate vector store queries and limit the prompt lenght to be the last 3 conversations to ensure it does not exceed context length.

There were also countless other Twitters and other media posts I read through to make this work. Two posts I would shout out here for giving me good direction: https://hamel.dev/notes/llm/03_inference.html and https://oobabooga.github.io/blog/posts/perplexities/ ultimately were the reason the app was using Exllama rather than other inference framework due to its speed and better perplexity than AutoGPTQ.

SharePoint API component to download documents is not yet implemented but not a complex piece. 

# Environment Setup (default for GPU and Windows set up unless stated otherwise)
In order to set your environment up to run the code here, before we install python packages dependancies, please ensure you have installed the following:

1. CUDA toolkit 11.8 (please ensure this is the only CUDA installed on you device, if not, first uninstall the current version and install 118. Trust me it would save you hours)

2. Install MSVC 2022. You can choose to install the whole Visual  Studio 2022 IDE, or alternatively just the Build Tools for Visual Studio 2022 package (make sure Desktop development with C++ is ticked in the installer), it doesn't really matter which.

3. Ensure that the system PATH variable has the cl.dll location added (it would be something like C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.36.32532\bin\Hostx64\x64). Also would be a good idea to double check and make sure the 2 CUDA paths were added (likely to be something like C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin and C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\libnvvp)

4. Ensure you install Docker desktop so you could run docker image (you would need it to run Qdrant server). When you attempt to install it, it should help you set up with Ubuntu distributable

(At this point I can't help but wonder whether it's more benefitial to just move to Linux and create a Docker image for all of this. Let's put it in the backlog...)

Now we are finally ready to install all (python packages) requirements:

```shell
pip install -r requirements.txt
```

One important thing to note regarding llama-cpp-python, if you would like to run with both GPU and CPU, best to follow the guide on GitHub page of the package and install CuBLAS version.

Now for the more complicated installation that requires running pip with wheels and custom set up (I know, but that's the price for GPU computing and shiny new toys):

1. cd to dependancies folder, and pip install all of the files
2. Copy and paste the py script in the langchain custom to python installation folder Lib\site-packages\langchain\llms (I had to hack a Exllama module to make Langchain work with Exllama, there could be all sort of bugs so be aware)

Then, download the LLM model and place it in a directory of your choice. Just remember to set up in the .env.

Rule of thumb for downloading models is, GPTQ for Exllama and GGML for Llama.cpp

Edit the variables in .env appropriately (this was originally set in PrivateGPT repo, since then it has been modified and more applicable to Llama.cpp scenario).
```
MODEL_TYPE: supports LlamaCpp only now
PERSIST_DIRECTORY: is the folder you want your vectorstore in (not applicable anymore as its set in Qdran server start-up now)
MODEL_PATH: Path to your LLM binary file/safetensor file
MODEL_N_CTX: Maximum token limit for the LLM model
EMBEDDINGS_MODEL_NAME: SentenceTransformers embeddings model name (see https://www.sbert.net/docs/pretrained_models.html). Currently using intfloat/e5-large-v2, not the best for Q&A tbh but generally it's pretty good
MODEL_N_GPU: Specific to Llama.cpp, determine how many layers of the LLM would be offload to GPU for computation. The more it offloads to GPU, the more vRAM required but generally the better it performs.
MODEL_THREAD: Recommend to set to be the number of the CPU cores, although this is a pretty elusive setting that requires quite some experimenting to get it optimal. Bottom line however is to never exceed thread number of CPU
MODEL_DIR: Must set for Exllama running situation, set to be the folder location of the LLM model
for temperature, top_p and top_k setting, refer to this post: https://www.reddit.com/r/LocalLLaMA/comments/1343bgz/what_model_parameters_is_everyone_using/
```

Note: because of the way `langchain` loads the `SentenceTransformers` embeddings, the first time you run the script it will require internet connection to download the embeddings model itself.

## Instructions for ingesting your own dataset

Put any and all your files into the `source_documents` directory

The supported extensions are:

   - `.csv`: CSV,
   - `.docx`: Word Document,
   - `.enex`: EverNote,
   - `.eml`: Email,
   - `.epub`: EPub,
   - `.html`: HTML File,
   - `.md`: Markdown,
   - `.msg`: Outlook Message,
   - `.odt`: Open Document Text,
   - `.pdf`: Portable Document Format (PDF),
   - `.pptx` : PowerPoint Document,
   - `.txt`: Text file (UTF-8),

After you open Docker, use CMD to run the following: docker run -p 6333:6333 -v {the folder path where you want to store the vector database data}/qdrant_storage:/qdrant/storage qdrant/qdrant

Then run the following command to ingest all the data placed in source_document folder.

```shell
python ingest.py
```

Note: during the ingest process no data leaves your local environment. You could ingest without an internet connection, except for the first time you run the ingest script, when the embeddings model is downloaded.

## Ask questions to your documents, locally!
In order to spin up the app and ask a question, cd to the project folder where the app.py locates and run a command like:

```shell
streamlit run app.py
```

# Other System Requirements

## Python Version
To use this software, you must have Python 3.10 or later installed. Earlier versions of Python will not compile.

## C++ Compiler
If you encounter an error while building a wheel during the `pip install` process, you may need to install a C++ compiler on your computer.

### For Windows 10/11
To install a C++ compiler on Windows 10/11, follow these steps:

1. Install Visual Studio 2022.
2. Make sure the following components are selected:
   * Universal Windows Platform development
   * C++ CMake tools for Windows
3. Download the MinGW installer from the [MinGW website](https://sourceforge.net/projects/mingw/).
4. Run the installer and select the `gcc` component.
