##Streamlit interface for the assistant

#Using Streamlit UI
import streamlit as st
import os
from doc_parser import extract_text_from_pdf, extract_text_from_image
from responder import split_text, get_most_relevant_chunks, generate_answer
from dotenv import load_dotenv

#loading environment variables
load_dotenv()

st.set_page_config(page_title="IntelliDoc - Document Reader", layout="centered")
st.title("IntelliDoc : Document Text Extractor")

#upload file
uploaded_file=st.file_uploader("Upload a PDF or Image File", type=["pdf","png","jpg","jpeg"])

if uploaded_file:
    file_path=os.path.join("temp",uploaded_file.name)
    os.makedirs("temp",exist_ok=True)
    with open(file_path,"wb") as f:
        f.write(uploaded_file.read())
    st.success("File uploaded successfully!")
    if file_path.endswith(".pdf"):
        text=extract_text_from_pdf(file_path)
    else:
        text=extract_text_from_image(file_path)
        
    #display extracted text from pdfs and images
    st.subheader("Extracted Text:")
    st.text_area("Result",text,height=400)
    chunks=split_text(text)

    #Display the number of chunks the document is split into for semantic search
    st.caption(f"Document split into {len(chunks)} chunks for semantic search.")

    #Let the user ask a question about the document
    st.subheader("Ask a question about the document:")
    user_question=st.text_input("Enter your question here")
    if user_question:
        with st.spinner("Getting your answer soon:"):
            if len(chunks)<3:
                top_chunks=get_most_relevant_chunks(chunks,user_question,len(chunks))
            else:
                top_chunks=get_most_relevant_chunks(chunks,user_question)
                
            answer, score=generate_answer(top_chunks,user_question)

        #Displaying the answer to the question
        st.markdown("Answer to your question:")
        st.write(answer)

        #Self-evaluation of model using scores
        if score<0.3:
            st.warning("This answer may be unreliable. Please verify.")
        elif score<0.6:
            st.info("Medium confidence. Consider reviewing the document.")
        else:
            st.success("High confidence answer.")







