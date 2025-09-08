import os
import requests
import json
from typing import Literal
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, create_react_agent

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Global variables
llm = None
db = None
agent = None

def setup_llm():
    """Initialize the LLM based on available API keys"""
    try:
        if os.getenv("OPENAI_API_KEY"):
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model="gpt-4", temperature=0)
        elif os.getenv("ANTHROPIC_API_KEY"):
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0)
        elif os.getenv("GOOGLE_API_KEY"):
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
        else:
            raise ValueError("No API key found. Please set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY")
    except ImportError as e:
        print(f"Missing package: {e}")
        return None

def download_database():
    """Download the Chinook database"""
    if os.path.exists("Chinook.db"):
        print("Database already exists.")
        return True
    
    print("Downloading Chinook database...")
    try:
        url = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"
        response = requests.get(url)
        
        if response.status_code == 200:
            with open("Chinook.db", "wb") as file:
                file.write(response.content)
            print("File downloaded and saved as Chinook.db")
            return True
        else:
            print(f"Failed to download the file. Status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error downloading database: {e}")
        return False

def setup_database():
    """Set up the database connection"""
    try:
        db = SQLDatabase.from_uri("sqlite:///Chinook.db")
        print(f"Database connected - Dialect: {db.dialect}")
        print(f"Available tables: {db.get_usable_table_names()}")
        return db
    except Exception as e:
        print(f"Error setting up database: {e}")
        return None

def create_agent(llm, db):
    """Create the SQL agent"""
    try:
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        tools = toolkit.get_tools()
        
        system_prompt = """
        You are an agent designed to interact with a SQL database.
        Given an input question, create a syntactically correct {dialect} query to run,
        then look at the results of the query and return the answer. Unless the user
        specifies a specific number of examples they wish to obtain, always limit your
        query to at most {top_k} results.
        You can order the results by a relevant column to return the most interesting
        examples in the database. Never query for all the columns from a specific table,
        only ask for the relevant columns given the question.
        You MUST double check your query before executing it. If you get an error while
        executing a query, rewrite the query and try again.
        DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
        database.
        To start you should ALWAYS look at the tables in the database to see what you
        can query. Do NOT skip this step.
        Then you should query the schema of the most relevant tables.
        """.format(
            dialect=db.dialect,
            top_k=10,
        )
        
        agent = create_react_agent(llm, tools, prompt=system_prompt)
        return agent
    except Exception as e:
        print(f"Error creating agent: {e}")
        return None

def initialize_app():
    """Initialize all components"""
    global llm, db, agent
    
    print("🚀 Initializing LangGraph SQL Agent API...")
    
    # Download database
    if not download_database():
        return False
    
    # Set up LLM
    llm = setup_llm()
    if not llm:
        print("❌ Failed to initialize LLM")
        return False
    
    # Set up database
    db = setup_database()
    if not db:
        print("❌ Failed to initialize database")
        return False
    
    # Create agent
    agent = create_agent(llm, db)
    if not agent:
        print("❌ Failed to create agent")
        return False
    
    print("✅ All components initialized successfully!")
    return True

@app.route('/')
def index():
    """Serve the frontend"""
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'llm_ready': llm is not None,
        'db_ready': db is not None,
        'agent_ready': agent is not None
    })

@app.route('/api/tables', methods=['GET'])
def get_tables():
    """Get available database tables"""
    if not db:
        return jsonify({'error': 'Database not initialized'}), 500
    
    try:
        tables = db.get_usable_table_names()
        return jsonify({'tables': tables})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/query', methods=['POST'])
def query_database():
    """Process natural language query"""
    if not agent:
        return jsonify({'error': 'Agent not initialized'}), 500
    
    try:
        data = request.get_json()
        question = data.get('question', '')
        
        if not question.strip():
            return jsonify({'error': 'Question cannot be empty'}), 400
        
        # Process the query
        messages = []
        final_response = ""
        
        for step in agent.stream(
            {"messages": [{"role": "user", "content": question}]},
            stream_mode="values",
        ):
            last_message = step["messages"][-1]
            messages.append({
                'type': last_message.__class__.__name__,
                'content': last_message.content,
                'tool_calls': getattr(last_message, 'tool_calls', [])
            })
            
            # Get the final AI response
            if hasattr(last_message, 'content') and last_message.content and not getattr(last_message, 'tool_calls', []):
                final_response = last_message.content
        
        return jsonify({
            'question': question,
            'answer': final_response,
            'messages': messages,
            'success': True
        })
        
    except Exception as e:
        print(f"Error processing query: {e}")
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/api/sample-questions', methods=['GET'])
def get_sample_questions():
    """Get sample questions"""
    questions = [
        "Which genre on average has the longest tracks?",
        "What are the top 5 most popular artists by number of tracks?",
        "How many customers are there in each country?",
        "What is the total revenue for each genre?",
        "Which artist has the most albums?",
        "What are the top 10 best-selling tracks?",
        "How many tracks are there in each genre?",
        "Which customer has spent the most money?",
        "What is the average track length by genre?",
        "How many employees work in each city?"
    ]
    
    return jsonify({'questions': questions})

if __name__ == '__main__':
    # Initialize the app
    if initialize_app():
        print("🌐 Starting Flask server...")
        print("📱 Open your browser and go to: http://localhost:8080")
        app.run(debug=True, host='0.0.0.0', port=8080)
    else:
        print("❌ Failed to initialize app. Please check your configuration.")