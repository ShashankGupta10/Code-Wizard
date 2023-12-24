import requests
import streamlit as st
from langchain.llms.together import Together
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import CohereEmbeddings
from langchain.chains import RetrievalQA
import os


def fetch_github_repo_contents(owner, repo, extensions, branch, path=''):
    url = f'https://api.github.com/repos/{owner}/{repo}/contents/{path}'
    params = {'ref': branch}
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"Failed to fetch repository contents. Status code: {response.status_code}")
        return []

    contents = response.json()
    files_with_extensions = []

    for content in contents:
        if content['type'] == 'file' and os.path.splitext(content['name'])[1] in extensions:
            print(content['name'])
            files_with_extensions.append(content['download_url'])
        elif content['type'] == 'dir':
            files_with_extensions.extend(fetch_github_repo_contents(owner, repo, extensions, branch, content['path']))

    return files_with_extensions


def get_text(owner, repo, extensions, branch):
    print(
        f"Fetching files with extensions {extensions} from {owner}/{repo}...")
    files_to_read = fetch_github_repo_contents(owner, repo, extensions, branch, '')
    print(files_to_read)
    all_text = ""
    if files_to_read:
        for file_url in files_to_read:
            response = requests.get(file_url)

            if response.status_code == 200:
                file_content = response.text
                all_text += file_content

            else:
                print(
                    f"Failed to read file from {file_url}. Status code: {response.status_code}")
        print(
            f"All files with extensions {extensions} have been saved to all_text")
    return all_text


def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n", ";"],
        length_function=len,
        chunk_size=1000
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(chunks):
    vector_store = FAISS.from_texts(chunks, CohereEmbeddings(
        cohere_api_key=st.secrets["COHERE_API_KEY"]))
    return vector_store


def main():
    st.title("Code Wizard 🧙")
    st.subheader("Generate code according to your coding conventions")
    owner = st.text_input("GitHub Repo Owner")
    repo = st.text_input("GitHub Repo Name")
    branch = st.text_input("GitHub Repo Branch")
    extensions = st.text_input(
        "File Extensions (seperated by comma ex: .py,.md,.js)").split(',')
    user_question = st.text_input(
        "User Question")
    prompt = f"You are a code generation bot.\n\
             You are given a user question.\n \
             Generate code for the user question.\n \
             The code should be in the language of the repo.\n \
             The code should be easy to understand and with comments as necessary.\n \
             The code should be well documented.\n \
             The code should be well tested.\n \
             The code should be well formatted.\n \
             The code should be well linted.\n \
             The code should be well optimized.\n \
             If you dont know how to generate code, just tell the user it is currently not in your capability to write this code.\n \
             If no langauge is specified, assume it is python.\n \
             The user question is: {user_question}\n"

    if st.button("Generate Code"):
        with st.spinner("Generating Code..."):
            llm = Together(
                model="Phind/Phind-CodeLlama-34B-v2",
                temperature=0.8,
                max_tokens=512,
                top_k=1,
                together_api_key=st.secrets["TOGETHER_API_KEY"]
            )
            text = get_text(owner, repo, extensions, branch)
            chunks = get_chunks(text)
            vector_store = get_vector_store(chunks)
            chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=vector_store.as_retriever(),
                verbose=True,
                chain_type="stuff",
            )
            answer = chain.run(prompt)
            st.markdown(answer)


if __name__ == "__main__":
    main()
