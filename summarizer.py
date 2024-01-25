# import dotenv
from dotenv import load_dotenv
load_dotenv()

# import langchain openai fastapi
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI


def summarizer(file_path):

    docs = TextLoader(file_path).load()

    # Define the prompt template
    prompt_template = """Summarize this: 
    {text}

    The summary should be concise, coherent, and capture the main points of the text, regardless of the document's length or topic.
    """
    prompt = PromptTemplate.from_template(prompt_template)

    # Initialize the model and output parser
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")  # Update to GPT 4 when finished with app development
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # Use StuffDocumentsChain if document length is within limits
    if len(docs[0].page_content) <= 4097:
        stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
        return stuff_chain.invoke(docs)["output_text"]

    # Map-Reduce method for longer documents
    else:
        # Map
        map_template = """The following is a set of documents
        {docs}
        Based on this list of docs, please identify the main themes 
        Helpful Answer:"""
        map_prompt = PromptTemplate.from_template(map_template)
        map_chain = LLMChain(llm=llm, prompt=map_prompt)

        # Reduce
        reduce_template = """The following is set of summaries:
        {docs}
        Take these and distill it into a final, consolidated summary of the main themes. 
        Helpful Answer:"""
        reduce_prompt = PromptTemplate.from_template(reduce_template)

        # Run chain
        reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

        # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
        combine_documents_chain = StuffDocumentsChain(
            llm_chain=reduce_chain, document_variable_name="docs"
        )

        # Combines and iteratively reduces the mapped documents
        reduce_documents_chain = ReduceDocumentsChain(
            combine_documents_chain=combine_documents_chain,
            collapse_documents_chain=combine_documents_chain,
            token_max=4000,
        )

        # Combining documents by mapping a chain over them, then combining results
        map_reduce_chain = MapReduceDocumentsChain(
            llm_chain=map_chain,
            reduce_documents_chain=reduce_documents_chain,
            document_variable_name="docs",
            return_intermediate_steps=False,
        )

        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000, chunk_overlap=0
        )
        split_docs = text_splitter.split_documents(docs)

        return map_reduce_chain.invoke(split_docs)["output_text"]


# Usage example
if __name__ == "__main__":
    file_path = "sample_texts/text.txt"
    output = summarizer(file_path)
    print(output)
    print("\n")
