# Recherche LLM dans un fichier data.txt

from dotenv import load_dotenv
import streamlit as st
from urllib.parse import urlparse, parse_qs
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback


def lire_fichier_data():
    with open("data.txt", "r") as f:
        texte = f.read()
    return texte


def main():
    load_dotenv()
    
    st.set_page_config(page_title="Assistant de l'Hôtel de la Sûre")
    
    # st.image("logo_couleur.png")
    st.header("Bienvenue sur notre assistant virtuel !")
    st.write("Je suis ravi de vous aider à répondre à toutes vos questions concernant notre hôtel, notre restaurant, notre spa et les environs. Nous sommes ici pour vous offrir une expérience inoubliable et enrichissante")
    st.write("Poser vos questions dans plusieurs langues. l'anglais, le français, le luxembourgoie, l'espagnol, l'allemand, l'italien, le néerlandais, le portugais, le danois, et encore d'autres...")
    
    # chargement data
    text = lire_fichier_data()
        
    # split into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
        )
    chunks = text_splitter.split_text(text)
    
    # create embeddings
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)
        
    # show user input
    user_question = st.text_input("Votre question:")
    if user_question:
        with st.spinner("Rédaction de la réponse, veuillez patienter..."):
            docs = knowledge_base.similarity_search(user_question)
            
            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=user_question)
                print(cb)
                
            st.write(response)


if __name__ == '__main__':
    main()
