##Handles embeddings, similarity search, and answering

#Using transformers - local LLM as openai requires licensing

from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import torch
from dotenv import load_dotenv
import os

load_dotenv()

# Load sentence-transformer model for semantic similarity
model=SentenceTransformer('all-MiniLM-L6-v2')
# Load QA pipeline from Hugging Face transformers
qa_pipeline=pipeline("question-answering", model="deepset/roberta-base-squad2",tokenizer="deepset/roberta-base-squad2")
def split_text(text,max_tokens=500):
    """
    Splits raw text into smaller chunks based on newlines.
    
    Args:
        text (str): Full text to be split.
        max_tokens (int): Approximate max length of each chunk.
    
    Returns:
        list: List of text chunks.
    """
    paragraphs=text.split("\n\n")
    chunks=[]
    current_chunk=""
    for para in paragraphs:
        if len(current_chunk)+ len(para)<max_tokens:
            current_chunk+=para+"\n\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk=para
    chunks.append(current_chunk.strip())
    return chunks
    
def get_most_relevant_chunks(text_chunks,question,top_k=3):

    """
    Ranks and retrieves the most relevant text chunks for a given question.
    
    Args:
        text_chunks (list): List of text chunks.
        question (str): User's input question.
        top_k (int): Number of top relevant chunks to return.
    
    Returns:
        list: Top-k relevant text chunks.
    """
    
    chunk_embeddings=model.encode(text_chunks, convert_to_tensor=True)
    question_embedding=model.encode(question, convert_to_tensor=True)
    similarities=util.pytorch_cos_sim(question_embedding, chunk_embeddings)[0]
    top_results=similarities.topk(top_k)
    return [text_chunks[i] for i in top_results.indices]

def generate_answer(context_chunks, question):

    """
    Generates an answer using the QA model from selected context chunks.
    
    Args:
        context_chunks (list): Chunks likely to contain the answer.
        question (str): The user's question.
    
    Returns:
        str: Best possible answer found and score
    """
    
    best_score=0
    best_answer="Sorry, I could not find a good answer in the document."
    for chunk in context_chunks:
        try:
            result=qa_pipeline({"context":chunk,"question":question})
            if result['score']>best_score and result['answer'].strip():
                best_score=result['score']
                best_answer=result['answer']
        except:
            continue
    return best_answer,best_score
