### mCDR Research Assistant
This Streamlit app uses GPT-4, Pinecone, and SentenceTransformers to synthesize marine carbon dioxide removal (mCDR) literature. It retrieves relevant document chunks and generates answers with citations.

### Live Demo
You can run this app on Streamlit Cloud or locally.

### Setup Instructions
1. Clone the repo
git clone https://github.com/your-username/mcdr-assistant.git
cd mcdr-assistant

### 2. Install dependencies
pip install -r requirements.txt

### 3. Set API keys

create a file called .streamlit/secrets.toml and add
OPENAI_API_KEY = "sk-..."
PINECONE_API_KEY = "pcsk-..."
PINECONE_INDEX_NAME = "ices-database-assistant"

### 4. Run the app
streamlit run app.py
