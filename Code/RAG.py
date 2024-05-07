# workflow
# knowledge base -> data loading -> node parsing -> create vector index
from __future__ import print_function
from googleFunctions import Form
import logging
import sys
import os
import glob
import pandas as pd
from llama_index.core import Settings
from llama_index.core import ( VectorStoreIndex, SimpleDirectoryReader, load_index_from_storage, StorageContext, ServiceContext)
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core.schema import MetadataMode
from llama_index.core.extractors import ( SummaryExtractor, QuestionsAnsweredExtractor, TitleExtractor, KeywordExtractor)
from llama_index.extractors.entity import EntityExtractor
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from dotenv import load_dotenv
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

class RAG:
    def __init__(self, args):

        self.llm = AzureOpenAI(
            model="gpt-35-turbo-16k",
            deployment_name="gpt-35-turbo-16k",
            api_key=args.api_key,
            azure_endpoint=args.api_endpoint,
            api_version=args.api_version,
        )

        self.embed_model = AzureOpenAIEmbedding(
            model="text-embedding-ada-002",
            deployment_name="text-embedding-ada-002",
            api_key=args.api_key,
            azure_endpoint=args.api_endpoint,
            api_version=args.api_version,
            embed_batch_size=50
        )
        
        load_dotenv('../Credentials/.env')

        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        credential = os.getenv("AZURE_OPENAI_API_KEY")
        azure_openai_api_version = "2024-04-01-preview"
        azure_openai_embedding_deployment = "text-embedding-ada-002"
        embedding_model_name = "text-embedding-ada-002"
        llm_model_name = "gpt-35-turbo-16k"
        api_type = "azure"

#TODO set to self.llm and self.embed model when you're done
        Settings.llm = llm
        Settings.embed_model = embed_model  

        if args.refresh == False:
            print("Loading Existing Index", file=sys.stderr)
            self.add_to_existing_RAG()

        if args.create == False:
            print("Creating New Index", file=sys.stderr)
            self.create_new_RAG()
        
        if args.metadata == False:
            print("Creating New Index", file=sys.stderr)
            self.create_new_metadataRAG()

        
    def parse_nodes_create_index(self, reader):
        documents = []
        for docs in reader.iter_data():
            documents.extend(docs)        

        # create the sentence window node parser w/ default settings
        print("Sentence Node Parsing", file=sys.stderr)
        sentence_node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=3,
            window_metadata_key="window",
            original_text_metadata_key="original_text"
        )

        nodes = sentence_node_parser.get_nodes_from_documents(documents)

        print("Defining Callback Manager", file=sys.stderr)
        callback_manager = CallbackManager()
        
        ctx_sentence = ServiceContext.from_defaults(llm=self.llm, embed_model=self.embed_model, node_parser=sentence_node_parser, callback_manager=callback_manager)

        index = VectorStoreIndex(nodes, service_context=ctx_sentence)
        
        return index
        
    def create_new_RAG(self):
        #data load to document object
        print("Loading Reader", file=sys.stderr)
        reader = SimpleDirectoryReader("./Data/coursedata_113113/", filename_as_id=True)

        documents = []
        for docs in reader.iter_data():
            documents.extend(docs)        

        # create the sentence window node parser w/ default settings
        print("Sentence Node Parsing", file=sys.stderr)
        sentence_node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=3,
            window_metadata_key="window",
            original_text_metadata_key="original_text"
        )

        nodes = sentence_node_parser.get_nodes_from_documents(documents)

        print("Defining Callback Manager", file=sys.stderr)
        callback_manager = CallbackManager()
        
        ctx_sentence = ServiceContext.from_defaults(llm=self.llm, embed_model=self.embed_model, node_parser=sentence_node_parser, callback_manager=callback_manager)

        index = VectorStoreIndex(nodes, service_context=ctx_sentence)
        
        ##save to persistent storage
        print("Saving to Persistent Storage", file=sys.stderr)
        index.storage_context.persist(persist_dir="./Data/ihp1_index")
        
        query_engine = index.as_query_engine(similarity_top_k=5)
        
    
    def add_to_existing_RAG(self):
#TODO: what the heck is going on in this function now that we've got two indexes AAAAHHHHHHH
        
        ##rebuild storage context
        retrieved_SC = StorageContext.from_defaults(persist_dir="./Data/ihp1_index/")

        ##load index
        retrieved_index = load_index_from_storage(retrieved_SC)

        reader = SimpleDirectoryReader("./Data/", filename_as_id=True)
        documents = []
        for docs in reader.iter_data():
            documents.extend(docs)             

        for d in documents:
            retrieved_index.refresh_ref_docs([d])

    def create_new_metadataRAG(self):
        text_splitter = TokenTextSplitter(
            separator=" ", chunk_size=512, chunk_overlap=128
        )

        extractors = [
            TitleExtractor(nodes=5, llm=llm),
            QuestionsAnsweredExtractor(questions=3, llm=llm),
            EntityExtractor(prediction_threshold=0.5),
            SummaryExtractor(summaries=["prev", "self"], llm=llm),
            KeywordExtractor(keywords=10, llm=llm)
        ]

        transformations = [text_splitter] + extractors

        reader = SimpleDirectoryReader("./Data/", recursive=True, filename_as_id=True, required_exts=[".pdf", ".docx", ".xlsx", ".pptx"])

        documents = []
        for docs in reader.iter_data():
            for filename in glob.glob('./Data/coursedata_*/course_file.json',recursive=True):
                #load pandas
                df = pd.read_json(filename)
                #course_file
                file_metadata = df.loc[df['filename'] == docs[0].metadata['file_name'].replace('_','+')]
                if file_metadata.empty != True:
                    file_metadata = file_metadata.squeeze().to_dict()
                    folder = file_metadata['folder_id']
                else:
                    pass
            for filename in glob.glob('./Data/coursedata_*/course_folder.json',recursive=True):
                #load pandas
                df = pd.read_json(filename)
                #course_file
                folder_metadata = df.loc[df['id'] == folder]
                if folder_metadata.empty != True:
                    folder_metadata = folder_metadata.squeeze().to_dict()
                else:
                    pass

            for doc in docs:
                doc.metadata.update(file_metadata)
                doc.metadata.update(folder_metadata)
            documents.extend(docs)

        pipeline = IngestionPipeline(transformations=transformations)

        nodes = pipeline.run(documents=documents)

        question_gen = LLMQuestionGenerator.from_defaults(
            llm=llm,
            prompt_template_str="""
                Follow the example, but instead of giving a question, always prefix the question 
                with: 'By first identifying and quoting the most relevant sources, '. 
                """
            + DEFAULT_SUB_QUESTION_PROMPT_TMPL,
        )
        
        llama_debug = LlamaDebugHandler(print_trace_on_end=True)
        callback_manager = CallbackManager([llama_debug])

        Settings.callback_manager = callback_manager

        # for i, (start_event, end_event) in enumerate(llama_debug.get_event_pairs(CBEventType.SUB_QUESTION)):
        #     qa_pair = end_event.payload[EventPayload.SUB_QUESTION]
        #     print("Sub Question " + str(i) + ": " + qa_pair.sub_q.sub_question.strip())
        #     print("Answer: " + qa_pair.answer.strip())
        #     print("====================================")

        sentence_node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=3,
            window_metadata_key="window",
            original_text_metadata_key="original_text"
        )

        nodes = sentence_node_parser.get_nodes_from_documents(documents)

        ctx = ServiceContext.from_defaults(llm=llm, embed_model=embed_model, node_parser=sentence_node_parser, callback_manager=callback_manager)

        print("LLM sees:\n",(nodes)[9].get_content(metadata_mode=MetadataMode.LLM))

        index = VectorStoreIndex(nodes, service_context=ctx)

        query_engine = index.as_query_engine(
            similarity_top_k=10,
            filter=[("<", "full_name", "2008-09-15")],
            sort_by="publication_date",
            llm=llm)

        
        index.storage_context.persist(persist_dir="./Data/index")

        result = query_engine.query(
            """
        What was the main financial regulation in the US before the 2008 financial crisis ?
        """
        )
        
        print(result.response)



        query_engine_tools = [
            QueryEngineTool(
                query_engine=engine,
                metadata=ToolMetadata(
                    name="course_documents",
                    description="course files from IHP1",
                ),
            ),
        ]

        query_engine = SubQuestionQueryEngine.from_defaults(
            query_engine_tools=query_engine_tools,
            question_gen=question_gen,
            service_context=ctx,
            use_async=True
        )

# response = query_engine.query(
#     """
#     Please create a 5 question review quiz on the functional anatomy of the thorax. Do not ask structural questions about which chapter contains which material, or any text from a photo/graph. Do not answer any questions with "all of the above" or "both" type answers. Please format the output like this:{"info":{"title": "", "description": "", "questions": [ { "question": "","type": "RADIO","options": [], "correct_answer": ""}]}}
#     """
# )

        response = query_engine.query(
            """
            Summarize the course content for coursedata_113113
            """
        )

        print(response)        


    def add_to_existing_metadataRAG(self):

        #Load index
        # retrieved_index = StorageContext.from_defaults(persist_dir="./data2/index/")
        # index = load_index_from_storage(retrieved_index)

    def extract(self, nodes):
        metadata_list = [
            {
                "custom": (
                    node.metadata["document_title"]
                    + "\n"
                    + node.metadata["excerpt_keywords"]
                )
            }
            for node in nodes
        ]
        return metadata_list

