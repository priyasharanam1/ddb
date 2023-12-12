from langchain import OpenAI 
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from waitress import serve
#from flask_restful import Resource, Api
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain import SQLDatabase
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv()
from langchain.prompts.prompt import PromptTemplate
# for testing
# hostname = os.getenv('hostname')
# username = os.getenv('username')
# password = os.getenv('password')
# database = os.getenv('database')
# os.environ["OPENAI_API_TYPE"] = "azure"
# os.environ["OPENAI_API_KEY"] = "9046fdf660364de6a64317150aaa1a54"
# os.environ["OPENAI_API_BASE"] = "https://azure-poc.openai.azure.com/"
# os.environ["OPENAI_API_VERSION"] = "2023-07-01-preview"

# #for deployment
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
# hostname = os.environ['hostname']
# username = os.environ['name']
# password = os.environ['password']
# database = os.environ['database']
# OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
# OPENAI_API_TYPE= os.environ["OPENAI_API_TYPE"] 
# OPENAI_API_BASE= os.environ["OPENAI_API_BASE"]
# OPENAI_API_VERSION= os.environ["OPENAI_API_VERSION"]

# llm =AzureChatOpenAI(deployment_name="saas-azure-gpt",model_name="gpt-35-turbo",temperature=0)
# os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(temperature=0,model_name='gpt-3.5-turbo')
# db_config = {
#     "user": username,
#     "password": password,
#     "host": hostname,
#     "port": 3306,
#     "database": database,
#     #"ssl_ca": "database-loader/debalesMYSQL_ssl_cert.pem",
#     "ssl_disabled": False
# }


#cnx = mysql.connector.connect(**db_config)

# SQLALCHEMY_DATABASE_URI= f'mysql+mysqlconnector://{db_config["user"]}:{db_config["password"]}@{db_config["host"]}:{str(db_config["port"])}/{db_config["database"]}'

SQLALCHEMY_DATABASE_URI=None
db=None
app = Flask(__name__)
CORS(app)

# Set up SQLAlchemy Engine
#engine = sqlalchemy.create_engine(SQLALCHEMY_DATABASE_URI)
#db = SQLDatabase.from_uri(SQLALCHEMY_DATABASE_URI)

# # Initialize models and chains
# llm = OpenAI(temperature=0)
# db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, top_k=100)
# chain = create_sql_query_chain(ChatOpenAI(temperature=0), db=db)

def table_question(input=""):
    #db = SQLDatabase.from_uri(SQLALCHEMY_DATABASE_URI)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent='zero-shot-react-description',
    handle_parsing_errors=True,
    )

    answer = agent_executor.run(input)
    return answer

table_question = Tool(
    name='table_question',
    func= table_question,
    description="For any question about the data cars, use car_data table. input is query"
)

tools =[table_question]

memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True,
)


#llm = ChatOpenAI(temperature=0,model_name='gpt-3.5-turbo')
conversational_agent = initialize_agent(
    agent='chat-conversational-react-description',
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3,
    early_stopping_method='generate',
    memory=memory,
    handle_parsing_errors=True,
)

# render the html page
@app.route('/')
def index():
    return render_template('index.html')


# API endpoint to ask a question and get data
@app.route('/query', methods=['POST'])
def process_query():
    data = request.get_json()
    if 'query' in data:
        query = data['query']
        response = conversational_agent.run(query)
        return jsonify({'response': response})+
    else:
        return jsonify({'error': 'Invalid request'}), 400
    
@app.route('/connect_db', methods=['POST'])
def connect_db():
    global SQLALCHEMY_DATABASE_URI
    global db
    #global db
    try:
        # Get data from the request
        data = request.get_json()

        # Extract values from the data
        database_link = data.get('databaseLink')
        database_username = data.get('databaseUsername')
        database_password = data.get('databasePassword')
        database_name = data.get('databaseName')

        # Your logic to connect to the database goes here
        # Replace the following print statements with your actual database connection logic

        # print(f"Connecting to the database with the following details:")
        # print(f"Database Link: {database_link}")
        # print(f"Database Username: {database_username}")
        # print(f"Database Password: {database_password}")
        # print(f"Database Name: {database_name}")
        db_config = {
            "user": database_username,
            "password": database_password,
            "host": database_link,
            "port": 3306,
            "database": database_name,
            #"ssl_ca": "database-loader/debalesMYSQL_ssl_cert.pem",
            "ssl_disabled": False
        }

        SQLALCHEMY_DATABASE_URI= f'mysql+mysqlconnector://{db_config["user"]}:{db_config["password"]}@{db_config["host"]}:{str(db_config["port"])}/{db_config["database"]}'

        db = SQLDatabase.from_uri(SQLALCHEMY_DATABASE_URI)

        print("connection done")
        return jsonify({'message': 'Successfully connected to the database'})

    except Exception as e:
        # Handle any exceptions that may occur during the database connection process
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=80)
