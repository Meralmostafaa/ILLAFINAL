from main import query_documents, generate_response
from flask import Flask, request, jsonify
import os
import requests

auth_token = os.getenv("AUTH_TOKEN")
website_id = os.getenv("CRISP_WEBSITE_ID")


app = Flask(__name__)

def chatbot_response(question):
    relevant_chunks = query_documents(question)
    answer = generate_response(question, relevant_chunks)
    return answer 

@app.route('/chatbot', methods=['POST']) 
def chatbot():
    try:
        data = request.json
        question = data.get("question")
        
        if not question:
            return jsonify({"error": "No question provided."}), 400

        # Generate chatbot response
        answer = chatbot_response(question)
        
        print(f"the answer is {answer}")
        
        return jsonify({"answer": answer})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/message_received', methods=['POST'])
def message_received():
    print("this is message received")
    try:
        response = request.json

        content = response.get("data").get("content")
        session_id = response.get("data").get("session_id")
        #message_text = data.get("message_text")
       
        # Process the message and generate a response
        response_text = chatbot_response(content)
        
        
        # Send the response back to Crisp
        send_response_to_crisp(session_id, response_text)
        
        return jsonify({"status": "success", "content":content, "our response": response_text}), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def send_response_to_crisp(session_id, response_text):
    url = f"https://api.crisp.chat/v1/website/{website_id}/conversation/{session_id}/message"
    headers = {
        "Authorization":  f"Basic {auth_token}",
        "Content-Type": "application/json",
        "X-Crisp-Tier" : "plugin"
    }
    payload = {
      "type": "text",
      "from": "operator",
      "origin": "chat",
      "content": response_text
    }
    response = requests.post(url, json=payload, headers=headers)
    return response

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  