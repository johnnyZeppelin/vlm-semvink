import os
import jieba
import re
from typing import List
import pdf
from loguru import logger
from magic_pdf.data.data_reader_writer import FileBasedDataWriter
from magic_pdf.pipe.UNIPipe import UNIPipe
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from tqdm import tqdm  

import nltk
nltk.download('punkt')

api_key = ""  # Add your actual API key
os.environ["OPENAI_API_KEY"] = api_key

# path to the Pharmacopoeia document
resume_folder = "./data/pharmacopoeia"
def extract_text_from_pdfs(folder_path):
    texts = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            with open(os.path.join(folder_path, filename), "rb") as file:
                reader = pdf.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                texts.append(text)
    return texts
def extract_text_from_markdown(folder_path):
    texts = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".md"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as file:
                text = file.read()
                texts.append(text)
    return texts

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9.,!?;:，。！？；：]', ' ', text)
    return text.strip()

def tokenize_texts(texts: List[str]) -> List[List[str]]:
    tokenized_texts = []
    for text in texts:
        cleaned_text = clean_text(text)
        words = []
        segments = jieba.cut(cleaned_text)
        words = [word for word in segments if word.strip()] 
        tokenized_texts.append(words)
    return tokenized_texts

def create_vector_store(tokenized_texts: List[List[str]], embeddings_model: OllamaEmbeddings) -> FAISS:
    try:
        processed_texts = [' '.join(tokens) for tokens in tokenized_texts]
        batch_size = 100
        vectors = []
        if FAISS.get_num_gpus():
            res = FAISS.StandardGpuResources()
            index = FAISS.index_cpu_to_gpu(res, 0, index)
        
        for i in tqdm(range(0, len(processed_texts), batch_size), desc="Creating vector store"):
            batch = processed_texts[i:i + batch_size]
            vector_store = FAISS.from_texts(
                texts=batch,
                embedding=embeddings_model,
                metadatas=[{"index": j} for j in range(i, i + len(batch))]
                )
            vectors.append(vector_store)
        if len(vectors) > 1:
            final_vector_store = vectors[0]
            for v in vectors[1:]:
                final_vector_store.merge_from(v)
        else:
            final_vector_store = vectors[0]
        final_vector_store.save_local("resume_vectors")

        return final_vector_store
    
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

# Extract text from PDFs or Markdown files
resume_texts = extract_text_from_markdown(resume_folder)
# resume_texts = extract_text_from_pdfs(resume_folder)
print("Done.")

# Tokenize the texts
tokenized_texts = tokenize_texts(resume_texts)
print(f"Done. Doc number: {len(tokenized_texts)}.")

for i, tokens in enumerate(tokenized_texts):
    print(f"Doc {i+1} has {len(tokens)} tokens in total.")

# Embedding the texts
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest",base_url='xxx')
print("Done.")

# Create the vector store
vector_store = create_vector_store(tokenized_texts, embeddings)
print("Done")

from langchain_community.llms import Ollama
llm = Ollama(model="llama3.3:70b", temperature=0.1, num_ctx=60000,base_url='xxx')

# Update imports
from langchain.chains import RetrievalQA  # Changed from create_retrieval_qa_chain
from langchain.prompts import PromptTemplate  # Import the necessary class

PROMPT_TEMPLATE = """
已知信息：{context}。
"你是一个资深的中医与中医药学专家，你具备管理大量药物处方和药材识别的能力；请注意我向你提供了很多《中华人民共和国药典》PDF文件中的内容，但每个PDF文件中包括药方、制备方法、药材特性等内容，请仔细识别各个信息。 \
现在需要你帮我分析每个中药或中成药的组成成分，只需要为我提供药典中此药品的各个组成成分药材名称即可，无需给出制备方法或用量等详细信息。然后帮我完成以下两个功能： \
1.当我给出一个中药名称和一组组成配方表时，给出其正确与否的答案。当基于我提供的药典信息，你认为问题中给出的配方不正确或者不匹配所提供药名时，只需回答“否”，反之，只需回答“是”即可。\
2.当我只给出一个中药名称时，请给出你认为此药物正确的配方表，只需要包含成分的药材名称即可，无需具体含量和制备过程。结果以markown形式给出。 \
    请务必注意有些药物的名称中可能含有类似中医药材但实际上不是药材的名称，切记不要望文生义； \
    有些药材经常用于制备药物，切记要根据所提供的内容谨慎地判断这些常见药材是否在当前给出的药物中也存在； \
    请仔细识别药物成分，不要重复给出药材名称。"

请回答以下问题：{question}
"""
# # Translation in English:
# Given the information: {context}
# "You are a senior expert in traditional Chinese medicine and pharmacology,
# with the ability to manage a large number of drug prescriptions and identify medicinal materials.
# Please note that I have provided you with extensive content from PDF files of the
# Pharmacopoeia of the People's Republic of China, each containing information such as prescriptions, preparation methods,
# and characteristics of medicinal materials. Please carefully identify each piece of information.
# Now, I need you to help me analyze the composition of each traditional Chinese drug or Chinese proprietary medicine.
# Simply provide me with the names of the constituent medicinal materials listed in the Pharmacopoeia for each drug,
# without including details such as preparation methods or dosages. Then, assist me in completing the following two tasks:
#   1. When I provide the name of a traditional Chinese drug and a set of constituent ingredients, give a correct or incorrect answer.
#      Based on the Pharmacopoeia information I have provided, if you determine that the given ingredients in the question are incorrect or
#      do not match the provided drug name, simply respond with "No". Otherwise, reply "Yes".
#   2. When I only provide the name of a traditional Chinese drug, please give what you believe to be the correct list of ingredients for this medicine.
#      Only include the names of the constituent medicinal materials, without specific quantities or preparation processes.
#   Please pay close attention to the following:
#     Some drug names may contain terms that resemble traditional Chinese medicinal materials but are not actual medicinal materials.
#       Do not interpret them literally.
#     Some medicinal materials are commonly used in drug preparation. Be cautious in determining whether these common materials are present
#       in the currently given medicine based on the provided content.
#     Carefully identify the drug components and avoid repeating the names of medicinal materials.
# Please answer the following question: {question}

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=PROMPT_TEMPLATE
)
chain_type_kwargs = {
    "prompt": prompt_template,
    "document_variable_name": "context",
}
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
    chain_type_kwargs=chain_type_kwargs,
)

def chat_loop():
    print("Input 'quit' or 'exit' to quit.")
    while True:
        user_input = input("\nInput your question: ").strip()
        if user_input.lower() in ['quit', 'exit']:
            print("See you!")
            break
        if user_input:
            try:
                # Get the response
                result = qa.run(user_input)
                print("\nAnswer: ")
                print(result)
            except Exception as e:
                print(f"Error: {str(e)}")
                continue

if __name__ == "__main__":
    chat_loop()
