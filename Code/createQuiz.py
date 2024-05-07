#!/usr/bin/env python3
# workflow
# knowledge base -> data loading -> node parsing -> create vector index

from __future__ import print_function
import logging
import sys
import os
from googleFunctions import Form
from RAG import RAG
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
import argparse, sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

def main(args):
    print("App Started", file=sys.stderr)
    CREDENTIALS_FILE = './Credentials/credentials.json'
    CLIENT_SECRETS_FILE = './Credentials/client_secret.json'

    SCOPES = ['https://www.googleapis.com/auth/forms.body']
    DISCOVERY_DOC = "https://forms.googleapis.com/$discovery/rest?version=v1"

    RAG(args)

    ##build storage context
    SC_sentence = StorageContext.from_defaults(persist_dir="./Data/index/")

    ##load index
    retrieved_index = load_index_from_storage(SC_sentence)

    #create query engine tool
    print("Creating Query Engine Tool", file=sys.stderr)
    query_engine = retrieved_index.as_query_engine(
        similarity_top_k=5,
        verbose=True,
        # the target key defaults to `window` to match the node_parser's default
        node_postprocessors=[
            MetadataReplacementPostProcessor(target_metadata_key="window")
        ],
    )

    
    # query engine
    # query = ("Fill out the blank in the following sentence from 'A pediatricians guide to climate change-informed primary care': 'Pediatricians can refer qualified at-risk families to _'")

    # query = ("What does Samantha Pullman say?'")

    # query = (
    #     'Please create a 5 question review quiz. Do not ask structural questions about which chapter contains which material, or any text from a photo/graph. Do not answer any questions with "all of the above" or "both" type answers. Please format the output like this:{"info":{"title": "", "description": "", "questions": [ { "question": "","type": "RADIO","options": [], "correct_answer": ""}]}}'
    # )
    
    query_response = query_engine.query(
        args.query
    )

    if args.quiz == False:
        # Google Quiz Creation
        form = Form(file_type='credentials', loginfile=CREDENTIALS_FILE,
                discovery_doc=DISCOVERY_DOC, scopes=SCOPES, sentence_response = query_response)

        link = form.get_link_to_form()
        print(link)

    else:
        # Print the response
        print(query_response, file=sys.stderr)

if __name__ == '__main__':
    parser=argparse.ArgumentParser()

    parser.add_argument("--create", action="store_false", help="Create a RAG from scratch using the contents of your local /Data directory")
    parser.add_argument("--refresh", action="store_false", help="Refresh/Update an existing RAG with new data")
    parser.add_argument("--api_key", type= str, required=True, action="store", help="Your OpenAI API key")
    parser.add_argument("--api_endpoint", type= str, required=True, action="store", help="Your Azure endpoint")
    parser.add_argument("--api_version", type= str, required=True, action="store", help="Your API version")
    parser.add_argument("--api_type", type= str, required=True, action="store", help="Your API type")
    parser.add_argument("--query", type= str, required=True, action="store", help="The question you'd like to ask the model")
    parser.add_argument("--quiz", action="store_false", help="The question you'd like to ask the model")
    parser.add_argument("--metadata", action="store_false", help="Use the automated metadata extraction functionality")
    args=parser.parse_args()

    if (args.create and args.refresh) or not (args.create or args.refresh):
        raise ValueError('Please select one RAG method, --create or --refresh')
    
    main(args)