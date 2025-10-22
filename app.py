import os
import streamlit as st
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from openai import OpenAI


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ices-database-assistant")

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    st.error("Missing API keys. Please set OPENAI_API_KEY and PINECONE_API_KEY as environment variables.")
    st.stop()


model = SentenceTransformer('all-MiniLM-L6-v2')
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(name=PINECONE_INDEX_NAME)
client = OpenAI(api_key=OPENAI_API_KEY)


def retrieve_relevant_chunks(query, top_k=30):
    query_embedding = model.encode(query).tolist()
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return results["matches"]

def format_citation(meta):
    return f"{meta.get('title', 'Untitled')} by {meta.get('author', 'Unknown')} ({meta.get('publication_year', 'n.d.')})"

def truncate(text, max_chars=3000):
    return text[:max_chars]

def summarize_batch(batch):
    context = "\n\n".join(truncate(m["metadata"]["text"]) for m in batch if "text" in m["metadata"])
    prompt = f"""You are a scientific assistant. Summarize key research gaps in mCDR based on the following context. Cite specific papers and authors where possible.

Context:
{context}

Answer:"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def generate_response_with_citations(query, matches):
    batch_size = 5
    summaries = []
    citations = []

    for i in range(0, len(matches), batch_size):
        batch = matches[i:i+batch_size]
        summaries.append(summarize_batch(batch))
        for match in batch:
            citations.append(format_citation(match["metadata"]))

    final_prompt = f"""You are a scientific assistant. Cite all information in line as [Author, Year]. 
Go through all papers to form a full answer,
and state "More papers available" when there are more papers that you didn't have the memory to go through.
Based on the following summaries, synthesize a comprehensive answer to the question: {query}

Summaries:
{'\n\n'.join(summaries)}

Answer:"""
    final_response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": final_prompt}]
    )
    return final_response.choices[0].message.content + "\n\nSources:\n" + "\n".join(set(citations))

def check_for_hallucinations(query, response, retrieved_chunks, max_context_chars=6000):
    context = ""
    for chunk in retrieved_chunks:
        if "text" in chunk["metadata"]:
            next_chunk = chunk["metadata"]["text"]
            if len(context) + len(next_chunk) > max_context_chars:
                break
            context += "\n\n" + next_chunk

    prompt = f"""You are a scientific reviewer. Evaluate whether the following answer is fully supported by the context.
Query: {query}

Answer:
{response}

Context:
{context}

Does the answer contain any unsupported claims, hallucinations, or fabricated citations? If so, list them. Otherwise, say 'Fully grounded.'"""

    review = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return review.choices[0].message.content

st.set_page_config(page_title="mCDR Research Assistant", layout="wide")
st.title("🔬 mCDR Research Assistant")
st.markdown("Ask a research question and get synthesized insights from your literature database.")

query = st.text_input("Enter your research question:")
run_review = st.checkbox("Run hallucination review")

if query:
    with st.spinner("Retrieving and synthesizing..."):
        docs = retrieve_relevant_chunks(query, top_k=30)
        if not docs:
            st.warning("No relevant documents found.")
        else:
            answer = generate_response_with_citations(query, docs)
            st.subheader("📄 Synthesized Answer")
            st.markdown(answer)

            if run_review:
                with st.spinner("Checking for hallucinations..."):
                    review = check_for_hallucinations(query, answer, docs)
                    st.subheader("🧠 Hallucination Review")
                    st.markdown(review)

st.markdown("---")
st.caption("Built by Brooke Beers using GPT-4, Pinecone, and SentenceTransformers. Data sourced from your mCDR literature corpus.")
