# workflow
# knowledge base -> data loading -> node parsing -> create vector index
from __future__ import print_function
import logging
import sys
import os
from googleFunctions import Form
from pprint import pprint
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceWindowNodeParser, get_leaf_nodes
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core import ( VectorStoreIndex, SimpleDirectoryReader, load_index_from_storage, StorageContext, ServiceContext)
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.callbacks import CallbackManager

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

def main():
    print("App Started", file=sys.stderr)
    CREDENTIALS_FILE = '/Credentials/credentials.json'
    CLIENT_SECRETS_FILE = '/Credentials/client_secret.json'

    SCOPES = ['https://www.googleapis.com/auth/forms.body']
    DISCOVERY_DOC = "https://forms.googleapis.com/$discovery/rest?version=v1"

    # define the azure openai endpoints (NOT openai base)
    api_key = ""
    azure_endpoint = ""
    api_version = ""
    api_type = ""

    # openai.api_type= api_type
    # openai.api_key = api_key

    #data load to document object
    print("Loading Reader", file=sys.stderr)
    reader = SimpleDirectoryReader("/Data/", recursive=True)
    documents = []
    for docs in reader.iter_data():
        print(f"Reading {docs}", file=sys.stderr)
        documents.extend(docs)

    #NODE PARSING AND INDEXING#

    # create the sentence window node parser w/ default settings
    print("Sentence Node Parsing", file=sys.stderr)
    sentence_node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text"
    )

    #define the llm as AzureOpenAI, not OpenAI - CRITICAL
    print("Defining LLM to AzureOpenAI", file=sys.stderr)
    llm = AzureOpenAI(
        model="gpt-35-turbo-16k",
        deployment_name="GPT-35-16K",
        api_key=api_key,
        azure_endpoint=azure_endpoint,
        api_version=api_version,
    )

    #create vector index  
    print("Defining Embed Model to AzureOpenAIEmbedding", file=sys.stderr)                                         
    embed_model_azure = AzureOpenAIEmbedding(
        model="text-embedding-3-large",
        deployment_name="text-embed-3L-rsg",
        api_key=api_key,
        azure_endpoint=azure_endpoint,
        api_version=api_version,
    )

    Settings.llm = llm
    Settings.embed_model = embed_model_azure

    print("Defining Callback Manager", file=sys.stderr)
    callback_manager = CallbackManager()
    
    print("Starting Vector Store Index", file=sys.stderr)
    sentence_index = VectorStoreIndex.from_documents(
    docs, embed_model=embed_model_azure, callback_manager=callback_manager
    )

    print("Saving to Persistent Storage", file=sys.stderr)
    ##save to persistent storage
    sentence_index.storage_context.persist(persist_dir="/Data/sentence_index")

    print("Rebuilding Storage Context", file=sys.stderr)
    ##rebuild storage context
    SC_retrieved_sentence = StorageContext.from_defaults(persist_dir="/Data/sentence_index/")

    ##load index
    retrieved_sentence_index = load_index_from_storage(SC_retrieved_sentence)

    #create query engine tool
    print("Creating Query Engine Tool", file=sys.stderr)
    sentence_query_engine = retrieved_sentence_index.as_query_engine(
        similarity_top_k=5,
        verbose=True,
        # the target key defaults to `window` to match the node_parser's default
        node_postprocessors=[
            MetadataReplacementPostProcessor(target_metadata_key="window")
        ],
    )

    # query engine
    question = ("Fill out the blank in the following sentence from 'A pediatricians guide to climate change-informed primary care': 'Pediatricians can refer qualified at-risk families to _'")
    # question = (
    #     'Please create a 5 question review quiz. Do not ask structural questions about which chapter contains which material, or any text from a photo/graph. Do not answer any questions with "all of the above" or "both" type answers. Please format the output like this:{"info":{"title": "", "description": "", "questions": [ { "question": "","type": "RADIO","options": [], "correct_answer": ""}]}}'
    # )
    sentence_response = sentence_query_engine.query(
        question
    )

    # Print the response
    print(sentence_response, file=sys.stderr)
    
    # # Google Quiz Creation
    # form = Form(file_type='credentials', loginfile=CREDENTIALS_FILE,
    #         discovery_doc=DISCOVERY_DOC, scopes=SCOPES, sentence_response = sentence_response)

    # link = form.get_link_to_form()
    # print(link)

if __name__ == '__main__':
    main()