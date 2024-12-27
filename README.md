# Wikipedia Semantic Search

A powerful semantic search engine for Wikipedia articles using OpenAI embeddings and natural language processing. This application allows users to search Wikipedia content using natural language queries and returns semantically relevant results.

## 🌟 Features

- Semantic search using OpenAI's text embeddings
- Real-time Wikipedia article fetching and processing
- Interactive Gradio web interface
- Structured logging for debugging and monitoring
- Rate limiting and retry mechanisms for API calls
- Automatic text chunking and processing
- Cosine similarity-based search results ranking

## 🛠 System Architecture

### High-Level System Overview

```mermaid
flowchart TD
    User[User/Client] -->|Search Query| WebUI[Gradio Web Interface]
    WebUI -->|Query Text| APP[Application Layer]
    APP -->|API Request| OpenAI[OpenAI API]
    APP -->|Article Request| Wikipedia[Wikipedia API]
  
    OpenAI -->|Embeddings| APP
    Wikipedia -->|Article Content| APP
  
    APP -->|Store| DB[(Embeddings Storage)]
    APP -->|Store| Meta[(Metadata Storage)]
  
    APP -->|Search Results| WebUI
    WebUI -->|Formatted Results| User
  
    style User fill:#f9f,stroke:#333
    style OpenAI fill:#bbf,stroke:#333
    style Wikipedia fill:#bfb,stroke:#333
    style DB fill:#fbb,stroke:#333
    style Meta fill:#fbb,stroke:#333
```

### Embedding Generation Process

```mermaid
flowchart TD
    Start[Start] -->|Input| FetchArticle[Fetch Wikipedia Article]
    FetchArticle -->|Raw Text| Clean[Clean & Preprocess Text]
    Clean -->|Processed Text| Chunk[Split into Chunks]
  
    Chunk -->|Text Chunks| TokenCount[Count Tokens]
    TokenCount -->|Chunks <= 1000 tokens| Embed[Generate Embeddings]
  
    Embed -->|Vector Data| Store[Store Embeddings]
    Store -->|Metadata| SaveMeta[Save Article Metadata]
  
    Embed -->|Error| Retry[Retry Logic]
    Retry -->|Retry Request| Embed
  
    SaveMeta --> UpdateIndex[Update Search Index]
    UpdateIndex --> End[End]
  
    style Start fill:#f9f,stroke:#333
    style Embed fill:#bbf,stroke:#333
    style Store fill:#bfb,stroke:#333
    style UpdateIndex fill:#fbb,stroke:#333
```

### Search Process Flow

```mermaid
flowchart TD
    Query[User Query] -->|Input Text| Process[Process Query]
    Process -->|Clean Text| GenEmbed[Generate Query Embedding]
  
    GenEmbed -->|Vector| LoadEmbed[Load Stored Embeddings]
    LoadEmbed -->|Vectors| Similarity[Calculate Cosine Similarity]
  
    Similarity -->|Scores| Rank[Rank Results]
    Rank -->|Top K| Format[Format Results]
  
    Format -->|JSON| Metadata[Add Metadata]
    Metadata -->|Enhanced Results| Present[Present to User]
  
    LoadEmbed -->|Cache Miss| FetchNew[Fetch New Articles]
    FetchNew -->|New Content| UpdateEmbed[Update Embeddings]
    UpdateEmbed --> LoadEmbed
  
    style Query fill:#f9f,stroke:#333
    style GenEmbed fill:#bbf,stroke:#333
    style Similarity fill:#bfb,stroke:#333
    style Present fill:#fbb,stroke:#333
```

### Error Handling and Retry Mechanism

```mermaid
flowchart TD
    Request[API Request] -->|Call| Check[Check Rate Limits]
    Check -->|OK| Process[Process Request]
    Check -->|Limited| Wait[Wait Period]
  
    Process -->|Success| Return[Return Result]
    Process -->|Error| Analyze[Analyze Error]
  
    Analyze -->|Retryable| Count[Check Retry Count]
    Count -->|Under Max| BackOff[Exponential Backoff]
    BackOff --> Request
  
    Count -->|Max Exceeded| Fail[Fail Request]
    Analyze -->|Fatal| Fail
  
    Wait -->|Time Elapsed| Request
  
    style Request fill:#f9f,stroke:#333
    style Process fill:#bbf,stroke:#333
    style BackOff fill:#bfb,stroke:#333
    style Fail fill:#fbb,stroke:#333
```

### Data Processing and Storage Flow

```mermaid
flowchart TD
    Input[Wikipedia Content] -->|Raw Text| Clean[Clean Text]
    Clean -->|Processed| Split[Split Sections]
    Split -->|Chunks| Remove[Remove References]
  
    Remove -->|Clean Text| Token[Tokenize]
    Token -->|Tokens| Batch[Create Batches]
  
    Batch -->|1000 Tokens| Embed[Generate Embeddings]
    Embed -->|Vectors| Store[Store Data]
  
    Store -->|Metadata| CSV[CSV Storage]
    Store -->|Vectors| Vector[Vector Storage]
  
    CSV -->|Index| Search[Search Index]
    Vector -->|Data| Search
  
    style Input fill:#f9f,stroke:#333
    style Embed fill:#bbf,stroke:#333
    style Store fill:#bfb,stroke:#333
    style Search fill:#fbb,stroke:#333
```

## 🛠️ Prerequisites

- Python 3.8+
- OpenAI API key
- Internet connection for Wikipedia access
- 📦 Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/wikipedia-semantic-search.git
   cd wikipedia-semantic-search
   ```
2. Install required packages:

   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project root:

   ```bash
   OPENAI_API_KEY=your_api_key_here
   ```

## 🚀 Usage

1. Start the application:

   ```bash
   python app.py
   ```
2. Access the Gradio interface at `http://localhost:7860`
3. Using the Search Interface:

   - Enter your search query
   - Adjust results count (1-10)
   - Click "Search"
   - View results with title, URL, summary, and relevance score

Example Queries:

- "What is quantum computing?"
- "Explain artificial intelligence"
- "History of space exploration"

## 🏗️ Project Structure

```
wikipedia-semantic-search/
├── app.py                 # Main application file
├── embedding_generator.py # Embedding generation
├── text_processor.py     # Text processing utilities
├── wiki_loader.py        # Wikipedia article fetching
├── utils.py             # Utility functions
├── main.py              # Entry point
├── requirements.txt     # Dependencies
└── data/               # Data storage
    ├── embeddings/     # Stored embeddings
    └── metadata.csv    # Article metadata
```

## 🔧 Technical Details

### Embedding Generation

- Uses OpenAI's `text-embedding-3-small` model
- Processes text in 1000-token batches
- Caches embeddings for performance

### Text Processing

- Splits articles into semantic chunks
- Removes references and unwanted sections
- Maintains hierarchical section structure

### Search Algorithm

1. Converts query to embedding
2. Computes cosine similarity
3. Ranks by similarity score
4. Returns top-k relevant sections

## 🔍 Advanced Usage

### Custom Search Parameters

```python
# Category search configuration
articles = fetch_wikipedia_articles(
    category_title="Artificial Intelligence",
    max_depth=2,
    max_articles=10
)

# Similarity search configuration
results = search_embeddings(
    query_embedding,
    embeddings_df,
    metadata,
    top_k=5
)
```

### Batch Processing

```python
from embedding_generator import generate_and_save_embeddings

categories = ["Artificial Intelligence", "Machine Learning", "Deep Learning"]
for category in categories:
    generate_and_save_embeddings(category, max_depth=2)
```

## 📊 Performance Notes

- Cached embeddings for faster response
- Rate-limited Wikipedia API calls
- Automatic retry for failed requests
- Structured logging system

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
