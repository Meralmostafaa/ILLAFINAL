from flask import Flask, request, render_template
import os
import time
from app import get_embedding, clean_text, add_embeddings_to_pinecone  

app = Flask(__name__)


UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


@app.route('/')
def upload_file():
    return render_template('upload.html')


import time

@app.route('/upload', methods=['POST'])
def upload_document():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    
    if file.filename == '':
        return 'No selected file'
    
    if file:
        
        filename = f"{int(time.time())}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        with open(filepath, 'r', encoding='utf-8') as f:
            document_text = f.read()

        add_embeddings_to_pinecone(document_text)

        return 'File uploaded, processed, and added to Pinecone successfully!'
    else:
        return 'Invalid file upload.'


if __name__ == "__main__":
    app.run(debug=True)
