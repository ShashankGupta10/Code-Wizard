import streamlit as st
from langchain.llms.together import Together
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import CohereEmbeddings
from langchain.chains import RetrievalQA
import os
import git

def fetch_files_in_directory(directory, extensions, files_list):
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        for i in extensions:
            if os.path.isfile(item_path) and item.endswith(i):
                files_list.append(item_path)
            elif os.path.isdir(item_path):
                fetch_files_in_directory(item_path, extensions, files_list)


def fetch_github_repo_contents(link, extensions, dir_name):
    try:
        git.Repo.clone_from(link, dir_name)
        print("Repo cloned successfully")
        files_with_extensions = []
        fetch_files_in_directory(
            dir_name, extensions, files_with_extensions)
        return files_with_extensions
    except git.GitError as e:
        print("Error:", e)
        st.error(
            "Failed to access github repository. Please make sure the repo is public and accessible...")


def get_text(link, extensions, dir_name):
    if not os.path.exists(dir_name):
        print(
            f"Fetching files with extensions {extensions} from {link}...")
        files_to_read = fetch_github_repo_contents(link, extensions, dir_name)
        print(files_to_read)
    else:
        files_to_read = []
        fetch_files_in_directory(dir_name, extensions, files_to_read)

    all_text = ""
    if files_to_read:
        for file_url in files_to_read:
            try:
                print(f"Reading file from {file_url}...")
                with open(file_url, 'r') as file:
                    file_content = file.read()
                    all_text += file_content
                print(f"File content from {file_url} successfully read.")
            except Exception as e:
                print(f"Failed to read file from {file_url}. Error: {str(e)}")
                break
        print(
            f"All files with extensions {extensions} have been saved to all_text")
        print(all_text)
    return all_text


def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n"],
        length_function=len,
        chunk_size=1000
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(chunks):
    if len(chunks) > 0:
        vector_store = FAISS.from_texts(chunks, CohereEmbeddings(
            cohere_api_key=st.secrets["COHERE_API_KEY"]))
        return vector_store
    st.info("No files found with the given extension.")
    return ""


def main():
    st.title("Code Wizard 🧙")
    st.subheader("Generate code according to your coding conventions")
    st.info("If your codebase is very extensive, please wait for some and let it go through all the files. It may take some time.")
    link = st.text_input("GitHub Repo Link")
    if link:
        dir_name = link.split("/")[4]
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
            text = get_text(link, extensions, dir_name)
            chunks = get_chunks(text)
            vector_store = get_vector_store(chunks)
            if vector_store != "":
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
