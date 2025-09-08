import os
import requests
from typing import Literal
from dotenv import load_dotenv

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, create_react_agent

# Load environment variables
load_dotenv()

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
        print("Install the required package: pip install langchain-openai langchain-anthropic or langchain-google-genai")
        return None

def download_database():
    """Download the Chinook database"""
    if os.path.exists("Chinook.db"):
        print("Database already exists.")
        return
    
    print("Downloading Chinook database...")
    url = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"
    response = requests.get(url)
    
    if response.status_code == 200:
        with open("Chinook.db", "wb") as file:
            file.write(response.content)
        print("File downloaded and saved as Chinook.db")
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")

def setup_database():
    """Set up the database connection and tools"""
    db = SQLDatabase.from_uri("sqlite:///Chinook.db")
    print(f"Dialect: {db.dialect}")
    print(f"Available tables: {db.get_usable_table_names()}")
    
    # Test query
    sample_query = "SELECT * FROM Artist LIMIT 5;"
    print(f'Sample output: {db.run(sample_query)}')
    
    return db

def create_prebuilt_agent(llm, db):
    """Create a simple prebuilt agent"""
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
        top_k=5,
    )
    
    agent = create_react_agent(llm, tools, prompt=system_prompt)
    return agent

def create_custom_agent(llm, db):
    """Create a custom agent with structured workflow"""
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()
    
    # Extract specific tools
    get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")
    get_schema_node = ToolNode([get_schema_tool], name="get_schema")
    
    run_query_tool = next(tool for tool in tools if tool.name == "sql_db_query")
    run_query_node = ToolNode([run_query_tool], name="run_query")
    
    # Define node functions
    def list_tables(state: MessagesState):
        tool_call = {
            "name": "sql_db_list_tables",
            "args": {},
            "id": "abc123",
            "type": "tool_call",
        }
        tool_call_message = AIMessage(content="", tool_calls=[tool_call])
        list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")
        tool_message = list_tables_tool.invoke(tool_call)
        response = AIMessage(f"Available tables: {tool_message.content}")
        return {"messages": [tool_call_message, tool_message, response]}
    
    def call_get_schema(state: MessagesState):
        llm_with_tools = llm.bind_tools([get_schema_tool], tool_choice="any")
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}
    
    generate_query_system_prompt = """
    You are an agent designed to interact with a SQL database.
    Given an input question, create a syntactically correct {dialect} query to run,
    then look at the results of the query and return the answer. Unless the user
    specifies a specific number of examples they wish to obtain, always limit your
    query to at most {top_k} results.
    You can order the results by a relevant column to return the most interesting
    examples in the database. Never query for all the columns from a specific table,
    only ask for the relevant columns given the question.
    DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
    """.format(
        dialect=db.dialect,
        top_k=5,
    )
    
    def generate_query(state: MessagesState):
        system_message = {
            "role": "system",
            "content": generate_query_system_prompt,
        }
        llm_with_tools = llm.bind_tools([run_query_tool])
        response = llm_with_tools.invoke([system_message] + state["messages"])
        return {"messages": [response]}
    
    check_query_system_prompt = """
    You are a SQL expert with a strong attention to detail.
    Double check the {dialect} query for common mistakes, including:
    - Using NOT IN with NULL values
    - Using UNION when UNION ALL should have been used
    - Using BETWEEN for exclusive ranges
    - Data type mismatch in predicates
    - Properly quoting identifiers
    - Using the correct number of arguments for functions
    - Casting to the correct data type
    - Using the proper columns for joins
    If there are any of the above mistakes, rewrite the query. If there are no mistakes,
    just reproduce the original query.
    You will call the appropriate tool to execute the query after running this check.
    """.format(dialect=db.dialect)
    
    def check_query(state: MessagesState):
        system_message = {
            "role": "system",
            "content": check_query_system_prompt,
        }
        tool_call = state["messages"][-1].tool_calls[0]
        user_message = {"role": "user", "content": tool_call["args"]["query"]}
        llm_with_tools = llm.bind_tools([run_query_tool], tool_choice="any")
        response = llm_with_tools.invoke([system_message, user_message])
        response.id = state["messages"][-1].id
        return {"messages": [response]}
    
    def should_continue(state: MessagesState) -> Literal[END, "check_query"]:
        messages = state["messages"]
        last_message = messages[-1]
        if not last_message.tool_calls:
            return END
        else:
            return "check_query"
    
    # Build the graph
    builder = StateGraph(MessagesState)
    builder.add_node(list_tables)
    builder.add_node(call_get_schema)
    builder.add_node(get_schema_node, "get_schema")
    builder.add_node(generate_query)
    builder.add_node(check_query)
    builder.add_node(run_query_node, "run_query")
    
    builder.add_edge(START, "list_tables")
    builder.add_edge("list_tables", "call_get_schema")
    builder.add_edge("call_get_schema", "get_schema")
    builder.add_edge("get_schema", "generate_query")
    builder.add_conditional_edges(
        "generate_query",
        should_continue,
    )
    builder.add_edge("check_query", "run_query")
    builder.add_edge("run_query", "generate_query")
    
    agent = builder.compile()
    return agent

def run_agent(agent, question):
    """Run the agent with a question and print results"""
    print(f"\n🤖 Running query: {question}")
    print("=" * 50)
    
    for step in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()

def main():
    print("🚀 Setting up LangGraph SQL Agent")
    
    # Download database
    download_database()
    
    # Set up LLM
    llm = setup_llm()
    if not llm:
        return
    
    # Set up database
    db = setup_database()
    
    print("\n📋 Choose agent type:")
    print("1. Prebuilt Agent (simple)")
    print("2. Custom Agent (structured workflow)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        agent = create_prebuilt_agent(llm, db)
        print("✅ Created prebuilt agent")
    else:
        agent = create_custom_agent(llm, db)
        print("✅ Created custom agent")
    
    # Example queries
    sample_questions = [
        "Which genre on average has the longest tracks?",
        "What are the top 5 most popular artists by number of tracks?",
        "How many customers are there in each country?",
        "What is the total revenue for each genre?",
    ]
    
    print("\n📝 Sample questions:")
    for i, q in enumerate(sample_questions, 1):
        print(f"{i}. {q}")
    
    print("\nType 'quit' to exit, or ask your own question:")
    
    while True:
        user_input = input("\n💬 Your question: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        elif user_input.isdigit() and 1 <= int(user_input) <= len(sample_questions):
            question = sample_questions[int(user_input) - 1]
            run_agent(agent, question)
        elif user_input:
            run_agent(agent, user_input)
        else:
            print("Please enter a question or 'quit' to exit.")

if __name__ == "__main__":
    main()