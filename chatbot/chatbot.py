from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain as DocLLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


class CPChatbot:
    def __init__(self, retriever, system_message):
        self.retriever = retriever.retriever
        self.system_message = system_message

        # Use Flan-T5 for better structured output
        model_id = "google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

        hf_pipeline = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
            device=-1  # CPU
        )

        self.llm = HuggingFacePipeline(pipeline=hf_pipeline)

        qa_prompt = ChatPromptTemplate.from_template(
            """
            You are a competitive programming chatbot. Help the user with the context below:
            {context}

            Using the above context, answer the following query:
            {question}
            """
        )

        question_prompt = ChatPromptTemplate.from_template(
            """
            Given the following conversation and a follow-up question, rephrase the question to be standalone.

            Chat History:
            {chat_history}

            Follow-up Question:
            {question}

            Standalone Question:
            """
        )

        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        self.question_generator_chain = LLMChain(llm=self.llm, prompt=question_prompt)

        doc_chain = StuffDocumentsChain(
            llm_chain=DocLLMChain(llm=self.llm, prompt=qa_prompt),
            document_variable_name="context"
        )

        self.qa = ConversationalRetrievalChain(
            retriever=self.retriever,
            combine_docs_chain=doc_chain,
            question_generator=self.question_generator_chain,
            memory=self.memory,
            return_source_documents=False
        )

    def chat(self, query):
        print(f"System Message: {self.system_message}")

        # DEBUG: Print retrieved documents
        docs = self.retriever.get_relevant_documents(query)
        print("\nRetrieved Documents:")
        for doc in docs:
            print(doc.page_content)

        results = self.qa.invoke({"question": query})
        print("\nFinal Answer:")
        print(results)
        return results