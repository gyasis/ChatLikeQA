from flask import Flask, render_template, request
import openai
import os
import platform
import chromadb
import langchain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import TokenTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ChatVectorDBChain
from langchain.document_loaders import GutenbergLoader

# Import the API-Key from the config file
from config import API_KEY

# Define the Flask app
app = Flask(__name__)

os.environ["OPENAI_API_KEY"] = API_KEY

# Load and process the Romeo and Juliet text data
def load_romeoandjuliet_data():
    # Define the URL of the Romeo and Juliet text data
    url = 'https://www.gutenberg.org/cache/epub/1513/pg1513.txt'

    # Load the data from the URL and split it into chunks
    loader = GutenbergLoader(url)
    data = loader.load()
    text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
    doc = text_splitter.split_documents(data)

    # Create the embeddings and vector database
    embeddings = OpenAIEmbeddings()
    
    data_folder = os.path.join(os.getcwd(), 'data')

    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    vectordb = Chroma.from_documents(doc, embeddings, persist_directory=data_folder)
   
    vectordb.persist()

    # Create the chatbot chain and return it
    chain = ChatVectorDBChain.from_llm(OpenAI(temperature=0, model_name="gpt-3.5-turbo"), vectordb, return_source_documents=True)
    return chain, [("Hello", "Hi there!")]

chain, chat_history = load_romeoandjuliet_data()

# Define the function to generate responses
def romeoandjuliet_qa(query, chat_history):
    result = chain({"question": query, "chat_history": chat_history})
    return result

# Define the Flask route for the chat interface
@app.route('/')
def chat():
    return render_template('chat.html')

# Define the Flask route to handle AJAX requests from the chat interface
@app.route('/get-response', methods=['POST'])
def get_response():
    global chat_history
    # Get the user's question from the AJAX request
    question = request.form['question']

    # Generate the chatbot's response
    result = romeoandjuliet_qa(question, chat_history)

    # Update the chat history with the user's input and the chatbot's response
    chat_history.append((question, result['answer']))

    # Return the chatbot's response
    return {'response': result['answer']}


# Add this to the app.py file
from werkzeug.utils import secure_filename
import io

# Define the allowed file types
ALLOWED_EXTENSIONS = {'txt', 'pdf'}

# Define the function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Define the route to handle file uploads
@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if a file was uploaded
    if 'file' not in request.files:
        flash('No file uploaded')
        return redirect(request.url)
    file = request.files['file']
    # Check if the file has a valid extension
    if file and allowed_file(file.filename):
        # Save the file to disk
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # Load the data from the file
        if filename.endswith('.txt'):
            with open(os.path.join(app.config['UPLOAD_FOLDER'], filename), 'r') as f:
                data = f.read()
        elif filename.endswith('.pdf'):
            with open(os.path.join(app.config['UPLOAD_FOLDER'], filename), 'rb') as f:
                data = io.StringIO(pdfminer.high_level.extract_text(f)).read()
        # Add the data to the vector database
        add_data_to_vectordb(data)
        flash('File uploaded successfully')
        return redirect(request.url)
    else:
        flash('Invalid file type')
        return redirect(request.url)
# Add this to the app.py file

def add_data_to_vectordb(data):
    # Split the data into chunks
    text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
    doc = text_splitter.split_documents(data)
    # Create the embeddings and vector database
    embeddings = OpenAIEmbeddings()
    data_folder = os.path.join(os.getcwd(), 'data')
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    vectordb = Chroma.from_documents(doc, embeddings, persist_directory=data_folder)
    vectordb.persist(append=True)
    
# Add this to the app.py file
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
