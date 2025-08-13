import os
import asyncio
import uuid
from typing import TypedDict, List, Optional, Dict, Any, Callable
from datetime import datetime, timezone
from functools import lru_cache
# from dotenv import load_dotenv
import openai
from llama_index.vector_stores.deeplake import DeepLakeVectorStore
from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores import VectorStoreQuery, MetadataFilters, ExactMatchFilter
from langgraph.graph import StateGraph, END
import traceback
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings

# Optional: FastAPI adapter (uncomment to use)
# from fastapi import FastAPI, HTTPException
# import uvicorn

# load_dotenv()

# ---------- Typed states ----------
class AgentState(TypedDict, total=False):
    user_input: str
    session: Dict[str, Any]
    scratchpad: List[str]
    selected_context: str
    compressed_history: str
    agent_context: str
    response: str

# ---------- Embedding helpers ----------
@lru_cache(maxsize=4096)
def cached_openai_embedding(text: str) -> tuple:
    """Synchronous wrapper (cached) for OpenAI embeddings.
       We return a tuple because lists are unhashable for caching; conversion handled by caller."""
    if not text or len(text) > 8192:
        return tuple()
    resp = openai.embeddings.create(
        model="text-embedding-3-small",
        input=[text]
    )
    return tuple(resp.data[0].embedding)

async def get_embedding(text: str) -> List[float]:
    # run the cached call in a threadpool so it doesn't block the event loop
    emb_tuple = await asyncio.to_thread(cached_openai_embedding, text)
    return list(emb_tuple)

# ---------- DeepLake store with batching + async wrappers ----------
class DeepLakePDFStore:
    def __init__(self, path: Optional[str] = None, commit_batch: int = 8, debug: bool = False):
        org_id = st.secrets.get("ACTIVELOOP_ORG_ID", "")
        path = path or f"hub://{org_id}/calum_v7"
        self.dataset_path = path
        self.commit_batch = commit_batch
        self.debug = debug or (os.getenv("DEBUG_CONVO") == "1")
        if self.debug:
            print(f"[DL] Init store path={path} batch={commit_batch}")

        # Unified read/write store
        # self.vector_store = DeepLakeVectorStore(
        #     dataset_path=path,
        #     token=st.secrets.get("ACTIVELOOP_TOKEN", ""),
        #     read_only=False
        # )

        self.vector_store = DeepLakeVectorStore(
            dataset_path=path,
            read_only=False
        )
        self.index = VectorStoreIndex.from_vector_store(self.vector_store)

    def debug_nodes(self, nodes: List[TextNode]):
        if not self.debug:
            return
        print(f"[DL] inserting {len(nodes)} node(s):")
        for n in nodes:
            txt = n.get_content() if hasattr(n, "get_content") else getattr(n, "text", "")
            print(f"  - id={getattr(n, 'id_', None)} len={len(txt)} meta={getattr(n, 'metadata', {})}")

    async def add_memory(self, agent: str, text: str):
        if not text:
            return
        if self.debug:
            print(f"[DL] add_memory queued text='{text[:60]}...'")
        
        # Chunk text
        chunks = self.chunk_text(text)
        nodes = []
        for chunk in chunks:
            node = TextNode(
                id_=str(uuid.uuid4()),
                text=chunk,
                metadata={
                    "agent": agent,
                    "type": "memory",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
            nodes.append(node)
        
        # Debug nodes
        self.debug_nodes(nodes)
        
        # Insert nodes via LlamaIndex
        try:
            self.index.insert_nodes(nodes)
            if self.debug:
                print("[DL] LlamaIndex insert_nodes done")
        except Exception as e:
            if self.debug:
                print(f"[DL] LlamaIndex insert_nodes error: {e}")

    async def flush(self):
        # No-op: using vector store managed commits
        return

    async def rag_query(self, query: str, top_k: int = 5, agent_id_filter: Optional[str] = None) -> List[str]:
        if not query:
            return []
        if self.debug:
            print(f"[DL] rag_query q='{query}' k={top_k} agent_filter={agent_id_filter}")

        retriever = self.index.as_retriever(similarity_top_k=top_k)

        agent_value = str(agent_id_filter if agent_id_filter is not None else "1")
        retriever.filters = MetadataFilters(filters=[ExactMatchFilter(key="agent", value=agent_value)])

        def _sync_retrieve(q):
            return retriever.retrieve(q)

        nodes = await asyncio.to_thread(_sync_retrieve, query)

        # Debugging: Print fetched results
        if self.debug:
            print(f"[DL] rag_query fetched {len(nodes)} results:")
            for i, node in enumerate(nodes):
                print(f"[{i+1}] ID: {node.id_}")
                print(f"    Text: {node.get_content()[:200]}")
                print(f"    Metadata: {node.metadata}")

        return [node.get_content() for node in nodes]

    async def debug_raw_query(self, text: str, k: int = 5, agent="1"):
        q_emb = Settings.embed_model.get_text_embedding(text)
        raw = self.vector_store.query(
            VectorStoreQuery(
                query_embedding=q_emb,
                similarity_top_k=k,
                filters=MetadataFilters(filters=[ExactMatchFilter(key="agent", value=str(agent))])
            )
        )
        print("RAW ids:", getattr(raw, "ids", None))
        print("RAW sims:", getattr(raw, "similarities", None))
        print("RAW metas:", getattr(raw, "metadatas", None))

    def chunk_text(self, text: str, chunk_size: int = 2000, overlap: int = 200) -> List[str]:
        if not text:
            return []
        chunks = []
        words = text.split()
        current_chunk = []
        for word in words:
            if sum(len(w) + 1 for w in current_chunk) + len(word) + 1 <= chunk_size:
                current_chunk.append(word)
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

# ---------- Orchestrator (LangGraph) ----------
class OrchestratedConversationalSystem:
    def __init__(self, session: Dict, agent: str = "1"):
        self.session = session
        self.graph = self._build_graph()
        self.agent = agent
        debug = session.get("debug", False) or (os.getenv("DEBUG_CONVO") == "1")
        self.store = DeepLakePDFStore(commit_batch=8, debug=debug)
        self.debug = debug
        # thresholds (configurable via session)
        self.max_turns = session.get("max_turns", 6)
        self.summarize_after_turns = session.get("summarize_after_turns", 6)

    def _summarize_history_sync(self, history: List[str]) -> str:
        """Sync wrapper for summarization call (used in to_thread)."""
        if not history:
            return ""
        # Small instructive prompt to summarizer LLM
        prompt = (
            f"You are {self.session.get('persona_name','Calum Worthy')}. "
            f"Summarize this conversation in 1-2 witty sentences:\n{' | '.join(history)}"
        )
        try:
            client = openai.OpenAI(
                api_key=self.session["api_key"],
                base_url=self.session.get("base_url", "https://api.openai.com/v1"),
            )
            resp = client.chat.completions.create(
                model=self.session.get("model_name", "gpt-4"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return ""

    async def _summarize_history(self, history: List[str]) -> str:
        return await asyncio.to_thread(self._summarize_history_sync, history)

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(AgentState)

        async def write_context_node(state: AgentState) -> AgentState:
            if self.debug:
                print("[NODE] write_context enter")
            st = dict(state)
            user_input = st.get("user_input")
            if not user_input:
                return st
            scratchpad = st.get("scratchpad", [])
            scratchpad.append(f"User: {user_input}")
            st["scratchpad"] = scratchpad[-self.max_turns:]
            # DO NOT add memory here!
            if self.debug:
                print("[NODE] write_context exit")
            return st

        def select_context_node(state: AgentState) -> AgentState:
            if self.debug:
                print("[NODE] select_context enter")
            st = dict(state)
            user_input = st.get("user_input", "")
            scratchpad_entries = st.get("scratchpad", [])[-8:]
            st["selected_context"] = {
                "memories_query": user_input,
                "recent_turns": scratchpad_entries
            }
            if self.debug:
                print("[NODE] select_context exit")
            return st

        async def resolve_context_node(state: AgentState) -> AgentState:
            if self.debug:
                print("[NODE] resolve_context enter")
            st = dict(state)
            sel = st.get("selected_context", {})
            q = sel.get("memories_query", "")
            memories = await self.store.rag_query(q, top_k=5, agent_id_filter="1")
            recent = sel.get("recent_turns", [])
            st["selected_context"] = f"Memories: {' | '.join(memories)}\nRecent: {' | '.join(recent)}"
            st["top_memory"] = memories[0] if memories else ""
            # NOW add the user input as memory
            await self.store.add_memory(self.agent, f"User: {q}")
            if self.debug:
                print("[NODE] resolve_context exit")
            return st

        async def compress_context_node(state: AgentState) -> AgentState:
            if self.debug:
                print("[NODE] compress_context enter")
            st = dict(state)
            if len(st.get("scratchpad", [])) >= self.summarize_after_turns:
                summary = await self._summarize_history(st.get("scratchpad", []))
                if summary:
                    await self.store.add_memory(self.agent, f"Summary: {summary}")
                    st["compressed_history"] = summary
            if self.debug:
                print("[NODE] compress_context exit")
            return st

        def isolate_context_node(state: AgentState) -> AgentState:
            if self.debug:
                print("[NODE] isolate_context enter/exit")
            st = dict(state)
            avatar_prompt = self.session.get("avatar_prompts", {}).get(self.session.get("avatar_id", ""), "")
            selected_context = st.get("selected_context", "")
            compressed_history = st.get("compressed_history", "")
            # Add debug print for context
            if self.debug:
                print("[DL] LLM context:\n", selected_context)
            st["agent_context"] = (
                f"You are Calum Worthy, a witty activist and actor. Answer as Calum, using your memories and context below.\n"
                f"Context:\n{selected_context}\n\nSummary:\n{compressed_history}"
            )
            return st

        async def llm_node(state: AgentState) -> AgentState:
            if self.debug:
                print("[NODE] llm enter")
            st = dict(state)
            # Otherwise, call LLM with context
            try:
                messages = [
                    {"role": "system", "content": st.get("agent_context", "")},
                    {"role": "user", "content": st.get("user_input", "")}
                ]
                def _call_llm():
                    client = openai.OpenAI(
                        api_key=self.session["api_key"],
                        base_url=self.session.get("base_url", "https://api.openai.com/v1"),
                    )
                    response = client.chat.completions.create(
                        model=self.session.get("model_name", "gpt-4"),
                        messages=messages,
                        temperature=self.session.get("temperature", 0.3)
                    )
                    return response

                resp = await asyncio.to_thread(_call_llm)
                answer = resp.choices[0].message.content.strip()
                scratchpad = st.get("scratchpad", [])
                scratchpad.append(f"{self.session.get('persona_name','Calum')}: {answer}")
                st["scratchpad"] = scratchpad[-self.max_turns:]
                st["response"] = answer
                asyncio.create_task(self.store.add_memory(self.agent, f"{self.session.get('persona_name','Calum')}: {answer}"))
            except Exception as e:
                if self.debug:
                    print("[LLM ERROR]", repr(e))
                    traceback.print_exc()
                st["response"] = "Oops — something went wrong. Try again?"
            if self.debug:
                print("[NODE] llm exit")
            return st

        # register nodes (mix sync & async nodes; LangGraph will handle)
        graph.add_node("write_context", write_context_node)
        graph.add_node("select_context", select_context_node)
        graph.add_node("resolve_context", resolve_context_node)
        graph.add_node("compress_context", compress_context_node)
        graph.add_node("isolate_context", isolate_context_node)
        graph.add_node("llm", llm_node)

        # edges
        graph.add_edge("write_context", "select_context")
        graph.add_edge("select_context", "resolve_context")
        graph.add_edge("resolve_context", "compress_context")
        graph.add_edge("compress_context", "isolate_context")
        graph.add_edge("isolate_context", "llm")
        graph.add_edge("llm", END)

        graph.set_entry_point("write_context")
        return graph.compile()

    async def run_turn(self, user_input: str, state: Optional[AgentState] = None) -> AgentState:
        if self.debug:
            print(f"[TURN] user_input='{user_input}'")
        state = state or AgentState(
            session=self.session,
            scratchpad=[],
            selected_context="",
            compressed_history="",
            agent_context="",
            response=""
        )
        state["user_input"] = user_input
        final_state = await self.graph.ainvoke(state)
        if self.session.get("force_sync_flush"):
            if self.debug:
                print("[TURN] force sync flush")
            await self.store.flush()
        else:
            asyncio.create_task(self.store.flush())
        return final_state

    # CLI convenience
    async def run_cli(self):
        print(f"{self.session.get('persona_name','Calum Worthy')} - (type 'exit' to quit)")
        state = AgentState(
            session=self.session,
            scratchpad=[],
            selected_context="",
            compressed_history="",
            agent_context="",
            response=""
        )
        while True:
            user_input = await asyncio.to_thread(input, "You: ")
            if user_input.strip().lower() == "exit":
                await self.store.flush()
                break
            new_state = await self.run_turn(user_input, state)
            # Force flush after first turn to ensure tensors exist
            await self.store.flush()
            print(f"{self.session.get('persona_name','Calum')}: {new_state.get('response','')}")
            state.update(new_state)

# ---------- Optional FastAPI adapter (simple) ----------
# app = FastAPI()
# conversational_system = OrchestratedConversationalSystem(session={
#     "api_key": os.getenv("OPENAI_API_KEY"),
#     "model_name": "gpt-4",
#     "base_url": "https://api.openai.com/v1",
#     "persona_name": "Calum",
#     "avatar_id": "calum",
#     "avatar_prompts": {"calum": "You are Calum Worthy, a witty activist and actor."}
# })
#
# @app.post("/chat")
# async def chat_endpoint(payload: Dict[str, str]):
#     user_input = payload.get("user_input", "")
#     if not user_input:
#         raise HTTPException(status_code=400, detail="user_input required")
#     state = AgentState(
#         session=conversational_system.session,
#         scratchpad=[]
#     )
#     new_state = await conversational_system.run_turn(user_input, state)
#     return {"response": new_state.get("response", "")}
#
# if __name__ == "__main__":
#     uvicorn.run("this_module:app", host="0.0.0.0", port=8000, reload=True)

# ---------- Run CLI if invoked directly ----------


if __name__ == "__main__":
    # at the top of convo.py (before you read env vars)
    
    

    import streamlit as st
    os.environ.setdefault("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY", ""))
    os.environ.setdefault("ACTIVELOOP_TOKEN", st.secrets.get("ACTIVELOOP_TOKEN", ""))
    os.environ.setdefault("ACTIVELOOP_ORG_ID", st.secrets.get("ACTIVELOOP_ORG_ID", ""))

    session = {
        "api_key": st.secrets.get("OPENAI_API_KEY", ""),
        "model_name": "gpt-4",
        "base_url": "https://api.openai.com/v1",
        "persona_name": "Calum",
        "avatar_id": "calum",
        "avatar_prompts": {"calum": "You are Calum Worthy, a witty activist and actor."},
        "temperature": 0.3,
        "debug": True,               # turn on verbose debug
        "force_sync_flush": False    # set True to wait every turn
    }
    # Initialize the embedding model BEFORE constructing the system
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=session.get("api_key")
    )
    conv = OrchestratedConversationalSystem(session=session)
    try:
        asyncio.run(conv.run_cli())
    except KeyboardInterrupt:
        asyncio.run(conv.store.flush())
        raise
