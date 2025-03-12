import os
import json
import ast
import re
import streamlit as st
from dotenv import load_dotenv
from pymongo import MongoClient
import pandas as pd
from langchain_openai import ChatOpenAI

# Set page config FIRST
st.set_page_config(
    page_title="AI MongoDB Query Generator",
    page_icon="🔍",
    layout="wide"
)

def sanitize_filter_string(filter_str: str) -> str:
    """Convert MongoDB-style syntax to valid Python dict syntax"""
    # Add quotes around operators
    filter_str = re.sub(r'\$(gt|gte|lt|lte|in|ne|eq)\b', r'"$\1"', filter_str)
    # Add quotes around unquoted keys
    filter_str = re.sub(r'(?<={|,)\s*(\w+)\s*:', r'"\1":', filter_str)
    # Convert to Python boolean values
    filter_str = filter_str.replace('true', 'True').replace('false', 'False')
    # Remove percentage signs
    filter_str = re.sub(r'(\d+)\s*%', r'\1', filter_str)
    # Fix trailing commas
    filter_str = re.sub(r',\s*}', '}', filter_str)
    return filter_str

def main():
    # Loading environment variables and schema
    load_dotenv()
    with open("database_schema.json") as f:
        SCHEMA = json.load(f)

    # Initialization LM Studio connection
    @st.cache_resource
    def get_llm():
        return ChatOpenAI(
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
            model="local-model",
            temperature=0.1,
            max_tokens=400
        )

    # Initialization of MongoDB connection
    @st.cache_resource
    def get_mongo():
        try:
            client = MongoClient(os.getenv("MONGO_URI"))
            return client[os.getenv("DB_NAME")][os.getenv("COLLECTION_NAME")]
        except Exception as e:
            st.error(f"🔴 MongoDB Connection Error: {str(e)}")
            st.stop()
    
    llm = get_llm()
    collection = get_mongo()

    # Sidebar with schema documentation
    with st.sidebar:
        st.header("📋 Database Schema")
        for field, details in SCHEMA["fields"].items():
            with st.expander(f"**{field}**"):
                st.markdown(f"""
                **Type:** `{details.get('type', 'string')}`  
                **Example:** `{details.get('example', 'N/A')}`  
                **Values:** `{', '.join(details.get('values', [])) if details.get('values') else 'Any'}`
                """)

    # Main interface
    st.title("🔍 AI-Powered MongoDB Query Generator")
    st.caption("Query your database using natural language powered by LM Studio")

    # User input
    question = st.text_area(
        "Ask your data question:",
        placeholder="Example: 'Show electronics products under $100 with rating above 4.5'",
        height=150
    )

    if st.button("🚀 Generate & Execute Query", use_container_width=True):
        if not question:
            st.error("Please enter a question first!")
            return

        # Generating query --------------------------------
        with st.status("🔧 Processing your request...", expanded=True):
            try:
                # Constructing AI prompt
                prompt = f"""Generate a MongoDB query for collection '{SCHEMA['collection']}'.
                
                **Schema:** {json.dumps(SCHEMA['fields'], indent=2)}
                
                **Question:** {question}
                
                **Rules:**
                1. Use proper MongoDB syntax starting with 'db.'
                2. Quote all field names and operators
                3. Use $ operators (e.g., "$gt" instead of >)
                4. Format numbers without percentage signs
                
                Return ONLY the plain query without code blocks."""

                # Getting LLM response --------------------------------
                response = llm.invoke(prompt)
                raw_response = response.content
                st.session_state.raw_response = raw_response
                
                # Extracting query using improved regex 
                query_match = re.search(
                    r'(db\..*?\.find\(.*?\))',
                    raw_response,
                    re.DOTALL
                )
                
                if not query_match:
                    raise ValueError("No valid query found in LLM response")
                
                query = query_match.group(1).strip()
                st.session_state.query = query
                
                st.subheader("Generated Query")
                st.code(query, language="javascript")

            except Exception as e:
                st.error(f"Query generation failed: {str(e)}")
                st.markdown(f"**Raw LLM Response:**\n```\n{raw_response}\n```")
                return

            # Query Execution 
            try:
                # Extracting filter criteria safely
                query_parts = query.split(".find(")
                if len(query_parts) < 2:
                    raise ValueError("Invalid query structure")

                filter_str = query_parts[1].rsplit(")", 1)[0]
                
                # Query sanitization --------------------------------
                filter_str = sanitize_filter_string(filter_str)
                st.write(f"Debug - Sanitized Filter: {filter_str}")
                
                # Convert to valid Python dict
                filter_dict = ast.literal_eval(filter_str)
                
                # Executing MongoDB query
                results = list(collection.find(filter_dict))
                
                if not results:
                    st.warning("No matching documents found")
                    return
                
                # Displaying results
                df = pd.DataFrame(results).drop('_id', axis=1)
                st.subheader(f"📊 Results ({len(df)} records)")
                st.dataframe(df, use_container_width=True)

                # Downloading button
                csv = df.to_csv(index=False).encode()
                st.download_button(
                    label="💾 Download CSV",
                    data=csv,
                    file_name="query_results.csv",
                    mime="text/csv"
                )

                # Saving query to log
                with open("queries_generated.txt", "a") as f:
                    f.write(f"Query: {query}\nResults: {len(df)} items\n\n")

            except Exception as e:
                st.error(f"Execution failed: {str(e)}\n\nFilter string: {filter_str}")
                st.markdown("**Query Debugging Tips:**\n"
                            "1. Check operator formatting (e.g., use \"$gt\" instead of $gt)\n"
                            "2. Verify all field names are quoted\n"
                            "3. Ensure proper JSON syntax")

if __name__ == "__main__":
    main()