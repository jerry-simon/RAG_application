from typing import List, Optional
from src.states.rag_state import RAGState

from langchain_core.documents import Document
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent_executor

from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun

class RAGNodes:

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self._agent = None


    def retrieve_docs(self, query: str) -> List[Document]:
        """Retrieve documents based on the query."""
        return self.retriever.retrieve(query)

    def generate_answer(self, state: RAGState) -> RAGState:
        """Generate an answer using the React agent based on the question in the state."""
        docs = self.agent.invoke(state.question)

        return RAGState(
            question=state.question,
            retrieved_docs=docs
        )
    
    def _build_tools(self) -> List[Tool]:
        """Build tools for the React agent."""
        def retriever_tool_func(query: str) -> str:
            docs: List[Document] = self.retriever.retrieve(query)
            
            if not docs:
                return "No relevant documents found."
            
            merged=[]
            for i, d in enumerate(docs[:8], start=1):
                meta = d.metadata if hasattr(d, 'metadata') else {}
                title = meta.get('title') or meta.get('source') or f"Document {i}"
                merged.append(f"[{i}] {title}:\n{d.page_content}\n")
            return "\n\n".join(merged)
        
        retriever_tool = Tool(
            name="Document Retriever",
            func=retriever_tool_func,
            description="Use this tool to retrieve documents relevant to the user's question."
        )

        wiki = WikipediaQueryRun(
            wikipedia_api_wrapper=WikipediaAPIWrapper(top_k_results=3,lang="en")
        )
        wikipedia_tool = Tool(
            name="Wikipedia",
            func=wiki.run,
            description="Useful for answering questions about general knowledge using Wikipedia."
        )
        return [retriever_tool, wikipedia_tool]
        
    
    def _build_agent(self):
        """Build the React agent with the necessary tools."""
        tools = self._build_tools()
        system_prompt = "You are a helpful AI assistant for answering questions based on retrieved documents."

        self._agent = create_react_agent_executor(self.llm, tools, prompt=system_prompt)

    def generate_answer(self, state: RAGState) -> RAGState:
        """Generate an answer using the React agent based on the question in the state."""
        if self._agent is None:
            self._build_agent()
        
        result = self._agent.invoke({"messages": [HumanMessage(content=state.question)]})

        messages = result.get("messages", [])
        answer: Optional[str] = None
        if messages:
            answer_msg = messages[-1]
            answer = getattr(answer_msg,"content", None)

        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=answer or "Could not generate an answer."
        )