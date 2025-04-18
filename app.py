import streamlit as st
from langchain.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from sqlalchemy import create_engine
import sqlite3
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import Ollama
from langchain.agents import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
import pandas as pd
import os
import tempfile
import re
from langchain_core.messages import SystemMessage

st.set_page_config(page_title="Chat with Excel Data", page_icon='📊')
st.title('Chat with your Excel Data Dynamically!!')

# Google API Key directly in code - consider moving to environment variable for security
DEFAULT_API_KEY = "AIzaSyDowIOAgzk-CMXEKXxQNsgGOuhJWNFiz7Q"

# Add model selection in sidebar
st.sidebar.header("Model Configuration")
model_option = st.sidebar.radio(
    "Choose AI Model",
    ["Google Gemini", "Ollama"]
)

# Ollama configuration if selected
if model_option == "Ollama":
    ollama_model = st.sidebar.selectbox(
        "Select Ollama Model", 
        ["llama2", "mistral", "codellama", "phi", "llama3"]
    )
    ollama_base_url = st.sidebar.text_input(
        "Ollama Base URL", 
        "http://localhost:11434"
    )

# Excel file upload
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload Excel file", type=['xlsx', 'xls'])

# Initialize the LLM based on user selection
if model_option == "Google Gemini":
    # Using the default API key
    api_key = DEFAULT_API_KEY
    llm = ChatGoogleGenerativeAI(
        google_api_key=api_key,
        model="gemini-1.5-flash",
        temperature=0,
        convert_system_message_to_human=True,
        streaming=True
    )
else:  # Ollama selected
    llm = Ollama(
        base_url=ollama_base_url,
        model=ollama_model,
        temperature=0
    )

@st.cache_resource(ttl='2h')
def configure_excel_db(excel_file):
    if not excel_file:
        st.error("Please upload an Excel file")
        st.stop()
    
    # Create a temporary SQLite database from the Excel file
    tmp_dir = tempfile.mkdtemp()
    tmp_db_path = os.path.join(tmp_dir, "excel_data.db")
    conn = sqlite3.connect(tmp_db_path)
    
    # Read all sheets from the Excel file
    excel_file = pd.ExcelFile(excel_file)
    sheet_names = excel_file.sheet_names
    
    # Process each sheet and add it to the database
    for sheet_name in sheet_names:
        # Get the dataframe from the current sheet
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        
        # Clean column names (remove spaces, special chars)
        df.columns = [col.lower().replace(' ', '_').replace('-', '_') for col in df.columns]
        
        # Create table name from sheet name (clean it for SQL)
        table_name = sheet_name.lower().replace(' ', '_').replace('-', '_')
        
        # Save this sheet to the SQLite database
        df.to_sql(table_name, conn, index=False, if_exists='replace')
        
    conn.close()
    
    # Show information about loaded sheets
    sheet_info = ", ".join(sheet_names)
    st.sidebar.success(f"Loaded {len(sheet_names)} sheets: {sheet_info}")
    
    return SQLDatabase(create_engine(f'sqlite:///{tmp_db_path}'))

# Custom QuerySQLDatabaseTool that cleans up markdown
class CustomQuerySQLDatabaseTool(QuerySQLDatabaseTool):
    def _run(self, query: str) -> str:
        """Clean query from markdown and execute it."""
        # Remove markdown SQL code blocks if present
        query = re.sub(r'```sql\s*|\s*```', '', query).strip()
        # Remove any trailing semicolons that might cause SQLite issues
        query = query.rstrip(';')
        
        try:
            return self.db.run(query)
        except Exception as e:
            return f"Error: {str(e)}"

# Custom toolkit that uses our custom query tool
class CustomSQLDatabaseToolkit(SQLDatabaseToolkit):
    def get_tools(self):
        tools = super().get_tools()
        
        # Replace the standard query tool with our custom one
        for i, tool in enumerate(tools):
            if isinstance(tool, QuerySQLDatabaseTool):
                tools[i] = CustomQuerySQLDatabaseTool(db=self.db, llm=self.llm)
                
        return tools

# Configure database from Excel file
if uploaded_file:
    db = configure_excel_db(uploaded_file)
else:
    st.info("Please upload an Excel file to begin")
    st.stop()

# Use our custom toolkit
toolkit = CustomSQLDatabaseToolkit(db=db, llm=llm)

# Add system message to encourage the model to correctly format its SQL
system_message = """You are an expert SQL agent that helps users query Excel data that has been converted to SQL tables. 

IMPORTANT: When writing SQL queries:
1. DO NOT include markdown formatting like ```sql or ``` around your queries
2. Keep your SQL syntax clean, without any markdown or formatting characters
3. DO NOT end your queries with semicolons
4. Make sure column names and table names are correctly referenced
5. Each Excel sheet has been converted to a separate table with the same name as the sheet

Remember to look at all available tables before attempting to query them. When presenting results to users, format tables nicely and explain what the data means in a clear way."""

agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    handle_parsing_errors=True,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    extra_prompt_messages=[SystemMessage(content=system_message)]
)

if 'messages' not in st.session_state or st.sidebar.button('Clear message history'):
    st.session_state['messages'] = [{'role': 'assistant','content':'How can I help you analyze your Excel data? You can ask questions in plain English.'}]
 
for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

user_query = st.chat_input(placeholder='Ask anything about your Excel data - you can query across all sheets!')

if user_query:
    st.session_state.messages.append({'role':'user','content':user_query})
    st.chat_message('user').write(user_query)

    with st.chat_message('assistant'):
        # Only use the StreamlitCallbackHandler for Google Gemini which supports streaming
        # Ollama integration doesn't support streaming in the same way
        if model_option == "Google Gemini":
            streamlit_callback = StreamlitCallbackHandler(st.container())
            response = agent.run(user_query, callbacks=[streamlit_callback])
        else:
            with st.spinner("Processing with Ollama..."):
                response = agent.run(user_query)
                
        st.session_state.messages.append({'role':'assistant','content':response})
        st.write(response)