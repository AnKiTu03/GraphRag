# Imports
import os
from typing import Tuple, List
from dotenv import load_dotenv
from IPython.display import display
from neo4j import GraphDatabase
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_community.vectorstores import Neo4jVector
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_fireworks import ChatFireworks, FireworksEmbeddings
from langchain_experimental.graph_transformers import LLMGraphTransformer
from yfiles_jupyter_graphs import GraphWidget
import warnings
from bs4 import GuessedAtParserWarning

warnings.filterwarnings("ignore", category=GuessedAtParserWarning)

# Load environment variables
load_dotenv()
os.environ["FIREWORKS_API_KEY"] = os.getenv("FIREWORKS_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Templates
History_template = """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question, in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

ans_template = """Answer the question based only on the following context:
{context}

Question: {question}
Use natural language and be concise.
Answer:"""

# Graph setup
graph = None
try:
    graph = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
    )
except ValueError as e:
    print(f"Could not connect to Neo4j database: {e}. Please ensure that the URL is correct.")

llm = ChatFireworks(model="accounts/fireworks/models/llama-v3p1-405b-instruct")
llm_transformer = LLMGraphTransformer(llm=llm)
default_cypher = "MATCH (s)-[r:!MENTIONS]->(t) RETURN s,r,t LIMIT 50"

# Entity Model
class Entities(BaseModel):
    """Identifying information about entities."""
    names: List[str] = Field(
        ..., description="All the person, organization, or business entities that appear in the text"
    )

def load_docs():
    """Load and split documents, then add them to the graph."""
    raw_documents = WikipediaLoader(query="Elizabeth I").load()
    text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
    documents = text_splitter.split_documents(raw_documents[:3])
    graph_documents = llm_transformer.convert_to_graph_documents(documents)
    if graph:
        graph.add_graph_documents(graph_documents, baseEntityLabel=True, include_source=True)
        # Create the full-text index 'entity' on the 'BaseEntity' label and 'id' property
        graph.query("""
        CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (n:BaseEntity) ON EACH [n.id]
        """)
    return None
def vectors_emb():
    vector_index = Neo4jVector.from_existing_graph(
        FireworksEmbeddings(),
        search_type="hybrid",
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding",
    )
    return vector_index


def showGraph(cypher: str = default_cypher):
    """Display graph widget for a given cypher query."""
    driver = GraphDatabase.driver(
        uri=NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
    )
    with driver.session() as session:
        widget = GraphWidget(graph=session.run(cypher).graph())
        widget.node_label_mapping = 'id'
        display(widget)
    return widget

def generate_full_text_query(input: str) -> str:
    """Generate a full-text query from input."""
    full_text_query = ""
    words = [el for el in remove_lucene_chars(input).split() if el]
    for word in words[:-1]:
        full_text_query += f" {word}~2 AND"
    full_text_query += f" {words[-1]}~2"
    return full_text_query.strip()

def structured_retriever(question: str) -> str:
    """Retrieve structured information related to the question."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are extracting organization and person entities from the text."),
            ("human", "Use the given format to extract information from the following input: {question}"),
        ]
    )
    entity_chain = prompt | llm.with_structured_output(Entities)

    result = []
    entities = entity_chain.invoke({"question": question})
    for entity in entities.names:
        response = graph.query(
    """
    CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
    YIELD node, score
    CALL (node) {
      MATCH (node)-[r:!MENTIONS]->(neighbor)
      RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
      UNION ALL
      MATCH (node)<-[r:!MENTIONS]-(neighbor)
      RETURN neighbor.id + ' - ' + type(r) + ' -> ' + node.id AS output
    }
    RETURN output LIMIT 50
    """,
    {"query": generate_full_text_query(entity)},
)

        result.extend([el['output'] for el in response])
    return "\n".join(result)

def retriever(question: str, vector_index):
    """Retrieve information based on the question from both structured and unstructured data."""
    print(f"Search query: {question}")
    structured_data = structured_retriever(question)
    unstructured_data = [el.page_content for el in vector_index.similarity_search(question)]
    final_data = f"""Structured data: {structured_data}
                     Unstructured data: {"#Document ".join(unstructured_data)}"""
    return final_data

def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
    """Format chat history for input into the model."""
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer

def convo(input, vector_index, chat_history):
    """Perform conversation handling, including condensing questions if there is chat history."""
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(History_template)
    _search_query = RunnableBranch(
        (
            RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(run_name="HasChatHistoryCheck"),
            RunnablePassthrough.assign(chat_history=lambda x: _format_chat_history(x["chat_history"]))
            | CONDENSE_QUESTION_PROMPT
            | ChatFireworks(temperature=0)
            | StrOutputParser(),
        ),
        RunnableLambda(lambda x: x["question"]),
    )
    
    prompt = ChatPromptTemplate.from_template(ans_template)
    chain = (
        RunnableParallel({"context": _search_query | (lambda q: retriever(q,vector_index)), "question": RunnablePassthrough()})
        | prompt
        | llm
        | StrOutputParser()
    )
    ans = chain.invoke({"question": input, "chat_history": chat_history})
    return ans

def generate_cypher_query(question: str) -> str:
    """Use LLM to generate a Cypher query based on the user question."""
    prompt_template = """
    Given the following question, generate a Cypher query to retrieve relevant information from a Neo4j graph.
    Assume all nodes are labeled as '__Entity__'.
    Question: {question}
    Cypher Query:
    """
    prompt = PromptTemplate.from_template(prompt_template)
    cypher_query = llm.invoke(prompt.format(question=question))
    return cypher_query

def main():
    vector = None
    load_data = input("Do you want to add a dataset? (yes/no): ").strip().lower()
    if load_data == "yes":
        vector = load_docs()
        print("Dataset added successfully.")
    
    vector = vectors_emb()
    chat_history = []

    while True:
        que = input("Enter the question (or type 'exit' to quit): ")
        if que.lower() == 'exit':
            print("Exiting chatbot. Goodbye!")
            break

        ans = convo(que, vector, chat_history)
        print(ans)
        chat_history.append((que, ans))

        # Ask if the user wants to generate and display a Cypher query
        generate_query = input("Would you like to generate a Cypher query for this question? (yes/no): ").strip().lower()
        if generate_query == "yes":
            cypher_query = generate_cypher_query(que)
            print(f"Generated Cypher Query:\n{cypher_query}")
            show_graph = input("Would you like to see the knowledge graph for this query? (yes/no): ").strip().lower()
            if show_graph == "yes":
                showGraph(cypher_query)

if __name__ == "__main__":
    main()