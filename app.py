import google.generativeai as genai
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from pydantic import BaseModel
from PIL import Image
import io
import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Initialize FastAPI app
app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load and process PDFs
pdf_folder = "ncertpdfs"
pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))

documents = []
for pdf_file in pdf_files:
    loader = PyPDFLoader(pdf_file)
    documents.extend(loader.load())

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(documents)

# Create vector store for retrieval
vectorstore = Chroma.from_documents(
    documents=docs, 
    embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})

# Initialize language model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 
    temperature=0, 
    max_tokens=None, 
    timeout=None
)

# Function to process doubts with text or image
def process_doubt(text=None, image=None):
    if not text and not image:
        return {"error": "Please provide a text query or an image."}
    
    if image:
        image = Image.open(io.BytesIO(image))
        model = genai.GenerativeModel("gemini-1.5-pro-vision")
        response = model.generate_content([image])
        text = response.text if response and hasattr(response, 'text') else "No text extracted from image."
    
    if text:
        retrieved_docs = retriever.get_relevant_documents(text)
    else:
        return {"error": "No valid text found to process."}
    
    if not retrieved_docs:
        return {"error": "No relevant document found."}
    
    context = retrieved_docs[0].page_content
    page_number = retrieved_docs[0].metadata.get("page", "Unknown")
    
    system_prompt = f"""
    You are an AI assistant specializing in answering doubts with clear and precise explanations.
    Given the following context, answer the user's question accurately:
    
    Context:
    {context}
    """
    
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content([system_prompt, f"User Query: {text}"])
    
    return {
        "pageNumber": page_number,
        "response": response.text if response and hasattr(response, 'text') else "No response received."
    }

# Define request model for JSON input
class AskRequest(BaseModel):
    text: str = None

# API endpoint to answer doubts with text or image
@app.post("/ask")
def ask(text: str = Form(None), file: UploadFile = File(None)):
    image_bytes = file.file.read() if file else None
    response = process_doubt(text=text, image=image_bytes)
    return response

# API endpoint to generate questions
@app.get("/generate_questions")
def generate_questions(
    chapterName: str = Query(..., description="Name of the chapter"),
    topicName: str = Query(..., description="Name of the topic"),
    userId: str = Query(..., description="User ID"),
    typeOfQuestions: str = Query("MCQ", description="Type of questions (MCQ, True/False, etc.)"),
    numberOfQuestions: int = Query(10, description="Number of questions to generate")
):
    system_prompt = (
        f"You are an AI assistant specialized in generating {typeOfQuestions} questions. "
        "Using the provided context, follow these instructions to generate questions:\n\n"
        "1. **Question Requirements:**\n"
        f"   - Create {numberOfQuestions} {typeOfQuestions} questions.\n"
        "   - Each question must have 4 options (A, B, C, D) if MCQ.\n"
        "   - Ensure the correct answers are present in the provided context.\n\n"
        "2. **Relevance Check:**\n"
        f"   - If the chapter name '{chapterName}' or topic '{topicName}' is not explicitly mentioned in the context, respond with 'Out of syllabus'.\n\n"
        "3. **Output Format:**\n"
        "   - Provide the output in JSON format.\n"
        "   - Each question and its options should be distinct properties within the JSON structure.\n\n"
        "4. **Make extremely difficult questions:**\n"
        "Input Context:\n"
        "{context}"
    )

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    response = rag_chain.invoke({"input": f"Chapter: {chapterName}, Topic: {topicName}"})
    return {"questions": response["answer"]}

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
