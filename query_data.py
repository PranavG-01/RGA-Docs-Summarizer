import argparse
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
import os
from openai import AzureOpenAI

from get_embedding_function import get_embedding_function

client = AzureOpenAI(
  azure_endpoint = os.getenv("https://pranavopenairesource.openai.azure.com/"), 
  api_key=os.getenv("237cb3c25a5d496287d689330c5bd2f9"),  
  api_version="2024-02-01"
)



CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=100)
    if len(results) == 0 or results[0][1] < 0.3:
        print(f"Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    print(context_text)
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    response = client.chat.completions.create(
        model="Pranav-LLM-Model", # model = "deployment_name".
        messages=[
            {"role": "system", "content": "Assistant is a large language model trained by OpenAI."},
            {"role": "user", "content": prompt}
        ]
    )

    print(response.choices[0].message.content)
    return(response.choices[0].message.content)



if __name__ == "__main__":
    main()