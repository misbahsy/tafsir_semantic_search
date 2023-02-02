# importing  important libraries
import streamlit as st
import cohere
import pinecone
import openai

st.header("This is a simple App to answer questions about the Quran")

#  storing api keys both cohere and pinecone
pinecone_api_key = 'cf643833-3fee-4700-9ac7-f0f90635a544'
cohere_api_key = 'z1DuayqafzKLEH4pOVHAPoDqKW5Gahqsut7mu00s'

# initializing cohere and pinecone
co = cohere.Client(cohere_api_key)

index_name = 'tafsir'
pinecone.init(pinecone_api_key, environment='us-west1-gcp')


# connect to index
index = pinecone.Index(index_name)

# defining the limit of the context
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

   
    # append contexts until hitting limit
    for i in range(1, len(contexts)):
        if len("\n\n---\n\n".join(contexts[:i])) >= limit:
            prompt = (
                " " +
                "\n\n---\n\n".join(contexts[:i-1]) + " ")
            break
        elif i == len(contexts)-1:
            prompt = (
                " " +
                "\n\n---\n\n".join(contexts) + " ")
    return prompt


query = st.text_area(":green[Enter Your :question: Question] :point_left:", help= " Ask the question about any topic in Quran",
             height=100, placeholder="Heyy, What are you waiting for, Enter your Question")

st.markdown("### if nothing is written in text box the above mentioned text is by default recommended to you")
    

# st.spinner("Searching for the answer")
query_with_contexts = retrieve(query)
query_with_contexts 



# get API key from top-right dropdown on OpenAI website
openai.api_key = "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"


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

complete(query_with_contexts)
# st.text(results)

# st.markdown(results)
