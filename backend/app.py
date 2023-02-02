from flask import Flask,request
import cohere
import pinecone
import openai 
import os
from dotenv import load_dotenv
load_dotenv()


api_key = os.environ.get('api_key')
cohere_api_key = os.environ.get('cohere_api_key')
openai.api_key = os.environ.get('openai_api_key')

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

co = cohere.Client(cohere_api_key)
index_name = 'tafsir'
pinecone.init(api_key, environment='us-west1-gcp')
index = pinecone.Index(index_name)


limit = 1600

def retrieve(query):
    xq = co.embed(
        texts=[query],
        model='multilingual-22-12',
        truncate='NONE'
    ).embeddings
    # search pinecone index for context passage with the answer
    xc = index.query(xq, top_k=3, include_metadata=True)
    contexts = [
        x['metadata']['text'] for x in xc['matches']
    ]

    # build our prompt with the retrieved contexts included
    prompt_start = (
        "Answer the Query based on the contexts, if it's not in the contexts say 'I don't know the answer'. \n\n"+
        "Context:\n"
    )
    prompt_end = (
        f"\n\nQuery: {query}\nAnswer in the language of Query, if Query is in English Answer in English."
    )
    # append contexts until hitting limit
    for i in range(1, len(contexts)):
        if len("\n\n---\n\n".join(contexts[:i])) >= limit:
            prompt = (
                prompt_start +
                "\n\n---\n\n".join(contexts[:i-1]) +
                prompt_end
            )
            break
        elif i == len(contexts)-1:
            prompt = (
                prompt_start +
                "\n\n---\n\n".join(contexts) +
                prompt_end
            )
    return prompt

def complete(prompt):
    # query text-davinci-003
    res = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        temperature=0,
        max_tokens=1000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    return res['choices'][0]['text'].strip()

@app.route('/api/predict', methods=['POST'])
def predict():
    query = request.json.get("query")
    query_with_contexts = retrieve(query)
    print(query_with_contexts)
    response = complete(query_with_contexts)

    return response
    
if __name__ == '__main__':
    app.run(debug=True)