# %%

# Import the required libraries and modules
from flask import Flask, render_template, request, flash, redirect
from werkzeug.utils import secure_filename
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import TokenTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ChatVectorDBChain
from langchain.document_loaders import GutenbergLoader
import chromadb
import os
import io
import pdfminer.high_level
from langchain.document_loaders import UnstructuredFileLoader
import tempfile
from langchain.document_loaders import UnstructuredFileLoader
import shutil
# %%


# Import the API-Key from the config file
from config import API_KEY

# Define the Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
os.environ["OPENAI_API_KEY"] = API_KEY

# Define the allowed file types
ALLOWED_EXTENSIONS = {'txt', 'pdf'}

# Define the function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Define the function to generate responses
def generate_response(query, chain, chat_history):
    # Call the chain function to generate a response
    result = chain({"question": query, "chat_history": chat_history})

    # Update the chat history with the user's input and the chatbot's response
    chat_history.append((query, result['answer']))

    # Return the chatbot's response
    return result

# %%
def clear_data_folder():
    data_folder = os.path.join(os.getcwd(), 'data')
    if os.path.exists(data_folder):
        for filename in os.listdir(data_folder):
            file_path = os.path.join(data_folder, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
# %%

def create_new_vectordb(filename, persist=True, overwrite_existing_db=False):
    global chain, chat_history
    # Read in the file data
    with open(os.path.join(app.config['UPLOAD_FOLDER'], filename), 'rb') as f:
        data = f.read()

    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(data)

    # Create the document loader and load the documents
    loader = UnstructuredFileLoader(tmp.name, strategy='fast')
    documents = loader.load()

    # Split the documents into pages and get the page content
    text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
    if isinstance(documents, str):
        # If documents is a string, wrap it in a list
        docs = text_splitter.split_documents([documents])
    else:
        # Otherwise, assume it's a list of documents
        docs = text_splitter.split_documents(documents)
        


    # Create the vector database
    embeddings = OpenAIEmbeddings()
    data_folder = os.path.join(os.getcwd(), 'data')
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    if overwrite_existing_db:
        #nucler option to removed contnet of data folder
        clear_data_folder()
        persist_directory = os.path.join(data_folder, filename.rsplit('.', 1)[0])
        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)
        os.makedirs(persist_directory)
    else:
        persist_directory = data_folder
    vectordb = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)

    # Create the chatbot chain and return it
    chain = ChatVectorDBChain.from_llm(OpenAI(temperature=0, model_name="gpt-3.5-turbo"), vectordb, return_source_documents=True)
    chat_history = []
    
    vectordb = Chroma.from_documents(docs,
                                     embeddings,
                                     persist_directory=data_folder)

    # Create the chatbot chain and return it
    chain = ChatVectorDBChain.from_llm(OpenAI(temperature=0, model_name="gpt-3.5-turbo"), vectordb, return_source_documents=True)
    chat_history = []

    # Persist the vector database if persist flag is set to True
    if persist: 
        vectordb.persist()

    return chain, chat_history
# %%

@app.route('/')
def home():
    return render_template('index.html')

# Define the route to the chat interface
@app.route('/chat', methods=['GET', 'POST'])
def chat():
    global chain, chat_history
    if request.method == 'POST':
        # Check the source of the request
        source = request.form.get('source')
        if source == 'index':
            # Load the persistent data
            chain, chat_history = load_persistent_data('vectordb')
        else:
            # Create a new vector database
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                chain, chat_history = create_new_vectordb(filename)
            else:
                flash('Invalid file type')
                return redirect(request.url)
    return render_template('chat.html')

# Define the route to the upload interface
@app.route('/upload')
def upload():
    return render_template('upload.html')


# Define the Flask route to handle AJAX requests from the chat interface
@app.route('/get-response', methods=['POST'])
# Define the Flask route to handle AJAX requests from the chat interface
@app.route('/get-response', methods=['POST'])

def get_response():
    global chat_history, chain
    # Get the user's question from the AJAX request
    question = request.form['question']

    # Generate the chatbot's response
    result = generate_response(question, chain, chat_history)

    # Update the chat history with the user's input and the chatbot's response
    chat_history.append((question, result['answer']))

    # Return the chatbot's response
    return {'response': result['answer']}

# Define the allowed file types
ALLOWED_EXTENSIONS = {'txt', 'pdf'}

# Define the function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Define the route to handle file uploads
@app.route('/upload-file', methods=['POST'])
def upload_file():
    global chain, chat_history
    # Check if a file was uploaded
    if 'file' not in request.files:
        flash('No file uploaded')
        return redirect(request.url)
    file = request.files['file']
    source = request.form.get('source', 'chat') # Get the source of the request (defaults to 'chat')
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
        if source == 'chat':
            # Add the data to the vector database
            add_data_to_vectordb(data, chain)
            chat_history = []
            chat_history.append(('INFO', f'Added data from {filename} to the vector database'))
        else: # source == 'index'
            # Create a new vector database
            chain, chat_history = create_new_vectordb(filename,overwrite_existing_db=True)
            chat_history.append(('INFO', f'Created new vector database from {filename}'))
        # Return the success message
        return redirect('/chat')
    else:
        flash('Invalid file type')
        return redirect(request.url)


# Define the function to add data to the vector database
def add_data_to_vectordb(data, chain):
    # Split the data into chunks
    text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
    doc = text_splitter.split_documents(data)
    # Create the embeddings and vector database
    embeddings = OpenAIEmbeddings()
    data_folder = os.path.join(os.getcwd(), 'data')
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    vectordb = Chroma.from_documents(doc, embeddings, persist_directory=data_folder)
    if os.path.exists(os.path.join(data_folder, 'vectordb')):
        vectordb.persist(append=True)
    else:
        vectordb.persist()
    # Create the new chatbot chain and return it
    chain = ChatVectorDBChain.from_llm(OpenAI(temperature=0, model_name="text-davinci-002"), vectordb, return_source_documents=True)


# Define the route to use the previous database
@app.route('/use-previous')
def use_previous():
    global chain, chat_history
    # Load the persistent data
    chain, chat_history = load_persistent_data('vectordb')
    # Return the chat interface
    return redirect('/chat')

# Define the route to start again with a new document
@app.route('/start-again')
def start_again():
    global chain, chat_history
    # Return the upload file interface
    return redirect('/upload-file')


# Run the Flask app
if __name__ == '__main__':
    # If running the app for the first time, create a new vector database
    data_folder = os.path.join(os.getcwd(), 'data')
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    if not os.path.exists(os.path.join(data_folder, 'persistent_data')):
        # If there is no persistent data or the user chooses to start fresh, prompt for a file upload
        app.run(debug=True, port=5050)
