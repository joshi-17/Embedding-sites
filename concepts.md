
| Term | Detailed Explanation |
|------|----------------------|
| Embeddings | Numerical representations of text in high-dimensional space (often 768 or 1536 dimensions) that capture semantic meaning and relationships between words or phrases. They allow machines to understand and process language by converting text into vectors that can be mathematically manipulated[1]. |
| Vectors | Ordered lists of numbers representing text in multi-dimensional space. For embeddings, each dimension corresponds to a latent feature of the text. Vectors enable mathematical operations like similarity comparisons between texts[1]. |
| Tokenization | The process of breaking down text into smaller units called tokens, which can be words, subwords, or characters. It's a crucial step in text processing that enables further analysis and embedding generation[4]. |
| Cosine Similarity | A metric used to measure the similarity between two non-zero vectors in an inner product space. In NLP, it's used to compare the similarity of embeddings by calculating the cosine of the angle between their vector representations. Values range from -1 to 1, with 1 indicating identical direction[5]. |
| Chunking | The technique of dividing large texts (like Wikipedia articles) into smaller, manageable sections. This improves processing efficiency and allows for more granular semantic search capabilities[6]. |
| Batching | Processing multiple items (e.g., text chunks) together in groups to optimize computational resources and API calls. In this project, embeddings are generated in batches of 1000 texts[7]. |
| Preprocessing | The cleaning and standardization of raw text data before further processing. This may include removing HTML tags, special characters, and standardizing format to ensure consistent and clean input for embedding generation[17]. |
| Indexing | The organization of embeddings and associated metadata in a structure that allows for efficient search and retrieval. This is crucial for quick semantic search operations[9]. |
| Caching | Storing frequently accessed data (like embeddings or search results) in fast-access memory to reduce computation time and API calls, thereby improving overall system performance[10]. |
| Metadata | Additional information about Wikipedia articles, such as titles, URLs, and summaries, stored alongside embeddings to provide context and enable rich search results[11]. |
| OpenAI API | An interface provided by OpenAI that allows developers to access various AI models, including the text-embedding-3-small model used in this project for generating text embeddings[12]. |
| Wikipedia API | A web service that provides programmatic access to Wikipedia content, allowing the retrieval of article text, metadata, and other information needed for the search engine[13]. |
| Gradio | An open-source Python library for building web-based interfaces for machine learning models. In this project, it's used to create an interactive user interface for the semantic search functionality[14]. |
| Rate Limiting | A technique to control the number of API requests made within a given time frame to prevent overloading servers and comply with usage policies. It's crucial for managing OpenAI and Wikipedia API usage[15]. |
| Backoff | An error handling strategy that implements progressively longer waits between retries of failed API calls. It helps manage temporary failures and improves system resilience[16]. |
| Text Cleaning | The process of removing unwanted elements (like HTML tags, special characters) and standardizing text format to prepare it for embedding generation and analysis[17]. |
| Section Splitting | Dividing Wikipedia articles into logical sections based on headings or content breaks. This allows for more precise semantic searching within specific parts of articles[6]. |
| Reference Removal | The process of cleaning citations, footnotes, and references from Wikipedia article text to focus on the main content for embedding generation[17]. |
| Semantic Search | A search technique that aims to understand the intent and contextual meaning of the user's query, rather than just matching keywords. It uses embeddings to find conceptually similar content[18]. |
| Similarity Ranking | The process of ordering search results based on their semantic similarity to the query. This typically involves calculating cosine similarity between query and document embeddings[5]. |
| Top-K Results | A configurable number of the most relevant search results returned to the user, typically ranked by similarity score[19]. |
| Text Embedding Model | A machine learning model (in this case, OpenAI's text-embedding-3-small) that converts text into high-dimensional vector representations capturing semantic meaning[1]. |
| Token Counting | The process of quantifying text length in terms of tokens, which is crucial for managing API limits and ensuring text fits within model constraints. The project uses tiktoken for this purpose[4]. |
| Web Interface | A browser-based user interface created with Gradio that allows users to interact with the semantic search system by entering queries and viewing results[14]. |
| Search Query | A natural language input from the user that expresses their information need. The system processes this query to find semantically relevant Wikipedia content[18]. |


Citations:
[1] https://www.ibm.com/think/topics/word-embeddings
[2] https://stackoverflow.blog/2023/11/09/an-intuitive-introduction-to-text-embeddings/
[3] https://neptune.ai/blog/vectorization-techniques-in-nlp-guide
[4] https://www.geeksforgeeks.org/nlp-how-tokenizing-text-sentence-words-works/
[5] https://www.engati.com/glossary/cosine-similarity
[6] https://www.nlpworld.co.uk/nlp-glossary/c/chunking/
[7] https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/
[8] https://exchange.scale.com/public/blogs/preprocessing-techniques-in-nlp-a-guide
[9] https://www.geeksforgeeks.org/role-of-search-indexer-in-information-retrieval-of-search-engine/
[10] https://www.getresponse.com/help/caching.html
[11] https://atlan.com/what-is-metadata/
[12] https://blog.hubspot.com/website/what-is-open-ai-api?uuid=856c91e8-f8bc-4ffb-bd6b-f06d118bbd6b
[13] https://dev.to/zuplo/what-is-the-wikipedia-api-how-to-use-it-and-alternatives-j4o
[14] https://www.niit.com/india/knowledge-centre/deep-learning-project
[15] https://tyk.io/learning-center/api-rate-limiting/
[16] https://bpaulino.com/entries/retrying-api-calls-with-exponential-backoff
[17] https://spotintelligence.com/2023/09/18/top-20-essential-text-cleaning-techniques-practical-how-to-guide-in-python/
[18] https://www.wallstreetprep.com/knowledge/semantic-search/
[19] http://www.ijceronline.com/papers/Vol8_issue6/I0806015459.pdf
[20] https://keylabs.ai/blog/understanding-precision-at-k-p-k/