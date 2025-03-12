import os
import re
from dotenv import load_dotenv
from pymongo import MongoClient
import pandas as pd
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]


valid_fields = list(collection.find_one().keys())

model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=200
)
llm = HuggingFacePipeline(pipeline=pipe)


prompt_template = """
You are an expert MongoDB query generator. Generate a query for the collection '{collection}' with fields: {fields}.

User Question: {question}
Respond ONLY with the valid MongoDB query (no explanations). Use ISO dates if needed.
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["question", "collection", "fields"]
)

query_chain = LLMChain(llm=llm, prompt=prompt)

def generate_query(question: str) -> str:
    fields = ", ".join(valid_fields)
    response = query_chain.run({
        "question": question,
        "collection": COLLECTION_NAME,
        "fields": fields
    })
    return re.sub(r'[\s`]+', ' ', response).strip()

def execute_query(query: str):
    try:
        if not query.startswith("db."):
            raise ValueError("Invalid query format.")
        filter_part = query.split(".find(")[1].split(")")[0]
        filter_dict = eval(filter_part)  
        results = list(collection.find(filter_dict))
        return results
    except Exception as e:
        print(f"Error executing query: {e}")
        return None

def main():
    while True:
        print("\nEnter your question (or 'exit' to quit):")
        question = input().strip()
        if question.lower() == 'exit':
            break
        
        query = generate_query(question)
        print(f"\nGenerated Query: {query}")
        
        with open("queries_generated.txt", "a") as f:
            f.write(f"{question}\n{query}\n\n")
        
        results = execute_query(query)
        if not results:
            continue
        
        df = pd.DataFrame(results)
        print("\nResults:")
        print(df.to_markdown(index=False))
        
        # Save to CSV
        save = input("\nSave to CSV? (y/n): ").strip().lower()
        if save == 'y':
            filename = input("Enter filename (e.g., test_case1.csv): ").strip()
            df.to_csv(filename, index=False)
            print(f"Saved to {filename}")

if __name__ == "__main__":
    main()