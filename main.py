from vector_store import get_vector_store
import config
from langchain_core.prompts import PromptTemplate

def main():
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    llm = config.llm  # Use the LLM from config

    prompt = PromptTemplate(
        template="""
          You are a helpful assistant.
          Answer ONLY from the provided transcript context.
          If the context is insufficient, just say you don't know.

          {context}
          Question: {question}
        """,
        input_variables=['context', 'question']
    )
    question = "is the topic of nuclear fusion discussed in this video? if yes then what was discussed"
    retrieved_docs = retriever.invoke(question)
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    final_prompt = prompt.invoke({"context": context_text, "question": question})
    # print(final_prompt)

    answer = llm.invoke(final_prompt)
    print(answer.content)




if __name__ == "__main__":
    main()