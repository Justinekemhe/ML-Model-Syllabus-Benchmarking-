import streamlit as st
from langchain.llms import OpenAI
from langchain.chains import LLMChain, SimpleSequentialChain, VectorDBQA
from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
import os
import PyPDF2
import docx
import tempfile

os.environ["OPENAI_API_KEY"] = "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"


def get_pdf_text(file_path):
    with open(file_path, "rb") as f:
        pdf_text = f.read()
    return pdf_text

def split_pdf_text(pdf_text, chunk_size=1000):
    chunks = []
    for i in range(0, len(pdf_text), chunk_size):
        chunks.append(pdf_text[i:i+chunk_size])
    return chunks

def get_embeddings(text_chunks):
    model_name = "sentence-transformers/bert-base-nli-mean-tokens"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    embeddings = []
    for chunk in text_chunks:
        input_ids = tokenizer.encode(chunk, return_tensors="pt")
        with torch.no_grad():
            output = model(input_ids)[0]
        embeddings.append(output.mean(dim=0).numpy())
    return np.array(embeddings)

def create_index(embeddings):
    index = IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index

def get_similar_chunks(query_embedding, embeddings, index, k=5):
    distances, indices = index.search(query_embedding.reshape(1,-1), k)
    return [(distances[0][i], indices[0][i]) for i in range(k)]

def get_response(query_embedding, text_chunks):
    embeddings = get_embeddings(text_chunks)
    index = create_index(embeddings)
    similar_chunks = get_similar_chunks(query_embedding, embeddings, index)
    response = ""
    for _, i in similar_chunks:
        response += text_chunks[i].decode("utf-8")
        response += "\n"
    return response

def extract_text(file, extension):
    if extension == '.txt':
        return file.getvalue().decode()
    elif extension == '.pdf':
        reader = PyPDF2.PdfFileReader(file)
        text = []
        for page_num in range(reader.numPages):
            text.append(reader.getPage(page_num).extractText())
        return ' '.join(text)
    elif extension == '.docx':
        doc = docx.Document(file)
        text = []
        for paragraph in doc.paragraphs:
            text.append(paragraph.text)
        return ' '.join(text)
    else:
        return None

def run_query(query, text):
    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.txt') as temp_file:
        temp_file.write(text)
        temp_file_path = temp_file.name

    loader = TextLoader(temp_file_path)
    index = VectorstoreIndexCreator().from_loaders([loader])
    result = index.query(query)

    os.unlink(temp_file_path)  


    return result
st.image('logo2.png', caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

st.markdown("<h3 style='text-align: center; color: grey;'>The Nelson Mandela African Institution of Science and Technology (NM-AIST)</h2>", unsafe_allow_html=True)

st.markdown("<h4 style='text-align: center; color: black;'>Language Model for Curriculum Benchmaking in Tanzania</h3>", unsafe_allow_html=True)
st.write("This tool uses cosine similarity function, to compare the embeddings of university curricula with job market requirements. it gives a measure of how well the curricula align with the job market demands.")
st.write("Based on the similarity scores, universities are ranked by their curricula's relevance to the job market requirements. It also analyze the results and provide insights and recommendations for improvements.")
st.sidebar.write("Uploaded Curriculum Here:")
uploaded_file = st.sidebar.file_uploader("Upload a file (.txt, .pdf, or .docx)", type=['txt', 'pdf', 'docx'])

if uploaded_file:
    file_ext = os.path.splitext(uploaded_file.name)[1]
    file_text = extract_text(uploaded_file, file_ext)

    if file_text:
        query = st.text_input("Ask a question about uploaded Curriculum:")
        if st.button('Submit'): 
            result = run_query(query, file_text)
            st.write(result)
    else:
        st.error("File format not supported or the file is empty.")
else:
    st.info("Please upload a file to get started.")
