# Chatbot for Competitive Programming
This project implements a BERT-based Competitive Programming Chatbot using embeddings, a FAISS Vector Store, and a Retrieval-Augmented Generation (RAG) Retriever to fetch relevant answers to coding queries. It combines semantic search over problem statements and editorials with an LLM to generate helpful, conversational answers for coding problem queries.

---

#Key Features
- Semantic Search using CodeBERT embeddings
- Fast similarity search with FAISS Vector Store
- Context-aware, conversational answers using Flan-T5 LLM
- Multi-turn conversations with LangChain Conversational Memory

---

#Project Structure:
```
chatbot/
├── chatbot.py           # Main chatbot logic with LangChain Conversational Retrieval
├── vectorstore.py       # FAISS vector store creation & management
├── retriever.py         # RAG retriever using FAISS
├── embeddings.py        # CodeBERT embedder for documents & queries
├── example.ipynb      # Example script to interact with the chatbot
└── GDG-CB4CP-Ass_1/
    ├── Problem Scraper/      # Problem statements (TXT files)
      ├──Problem_statement
      ├──Problem_metadata
      ├──Problem_scraper.py
    ├── Editorial Scraper/       # Editorial explanations (TXT files)
      ├──editorial_extraction.py
      ├──editorial
      ├──metadata_n_

  
```
---

#How it Works

1.Vector Store Creation
-Problem statements, editorials, and metadata are loaded
-Text is embedded using CodeBERT
-FAISS index is created for efficient similarity search

2.Query Handling
-User query is embedded using CodeBERT
-FAISS retrieves relevant documents
-Retrieved context + user query passed to Flan-T5 LLM
-LLM generates a conversational, context-aware response

3.Conversation Memory
-Chat history is maintained using LangChain's ConversationBufferMemory
-Supports multi-turn conversations with contextual continuity

---

#Limitations

-LLM Size: Flan-T5 Base is a small model; for complex reasoning or explanations, larger LLMs are recommended
-Data Dependence: The quality of chatbot answers depends heavily on the provided problem/editorial data

