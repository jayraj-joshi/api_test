from fastapi import FastAPI, HTTPException, Query, Depends, Form, UploadFile, File
from pydantic import BaseModel
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
import os
import glob
import asyncpg
import json
import re
from asyncio import to_thread
import random
from PIL import Image
import io
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("debug.log")]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://atomprep.vercel.app","https://atomrank.in","https://preview--quizchaos.lovable.app/"], 
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

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(documents)

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=1.0,
    max_tokens=None,
    timeout=None
)

import google.generativeai as genai
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

pool = None

async def get_db():
    async with pool.acquire() as connection:
        yield connection

@app.on_event("startup")
async def startup():
    global pool
    pool = await asyncpg.create_pool(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )

@app.on_event("shutdown")
async def shutdown():
    await pool.close()

def extract_json_from_markdown(response_text: str) -> str:
    pattern = r"```json\s*(.*?)\s*```"
    match = re.search(pattern, response_text, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        return json_str
    raise ValueError("No valid JSON found in the response")

def question_equals(q1, q2):
    if not isinstance(q1, dict) or not isinstance(q2, dict):
        return False
    
    # Safely compare questions
    q1_question = q1.get("question", "").lower().strip()
    q2_question = q2.get("question", "").lower().strip()
    
    # Handle answers that might be strings or booleans
    q1_answer = q1.get("answer")
    q2_answer = q2.get("answer")
    
    # Convert answers to strings for comparison if they’re not already
    if isinstance(q1_answer, bool):
        q1_answer = str(q1_answer).lower()
    elif isinstance(q1_answer, str):
        q1_answer = q1_answer.lower().strip()
    else:
        q1_answer = ""
        
    if isinstance(q2_answer, bool):
        q2_answer = str(q2_answer).lower()
    elif isinstance(q2_answer, str):
        q2_answer = q2_answer.lower().strip()
    else:
        q2_answer = ""
    
    # Log for debugging if types are unexpected
    if not isinstance(q1.get("answer"), (str, bool)) or not isinstance(q2.get("answer"), (str, bool)):
        logger.warning(f"Unexpected answer type: q1={q1.get('answer')}, q2={q2.get('answer')}")

    return q1_question == q2_question and q1_answer == q2_answer

def normalize_names(chapter_name: str, topic_name: str):
    return chapter_name.lower(), topic_name.lower()

@app.get("/generate_questions")
async def generate_questions(
    chapterName: str = Query(..., description="Name of the chapter"),
    topicName: str = Query(..., description="Name of the topic"),
    userId: str = Query(..., description="User ID"),
    typeOfQuestions: str = Query("MCQ", description="Type of questions (MCQ, True/False, etc.)"),
    numberOfQuestions: int = Query(10, description="Number of questions to generate"),
    db: asyncpg.Connection = Depends(get_db)
):
    chapterName, topicName = normalize_names(chapterName, topicName)
    logger.info(f"Request received: userId={userId}, chapterName={chapterName}, topicName={topicName}, typeOfQuestions={typeOfQuestions}, numberOfQuestions={numberOfQuestions}")

    # Step 1: Retrieve existing questions
    existing_questions_raw = await db.fetch(
        "SELECT question FROM user_questions WHERE user_id = $1 AND chapter_name = $2 AND topic_name = $3",
        userId, chapterName, topicName
    )
    existing_questions = [json.loads(q["question"]) for q in existing_questions_raw]
    logger.info(f"Existing questions fetched: {len(existing_questions)} questions")
    logger.debug(f"Existing questions content: {json.dumps(existing_questions, indent=2)}")

    # Step 2: Generate questions
    batch_size = numberOfQuestions * 3
    random_seed = random.randint(1, 10000)
    random_number = random.randint(1, 1000)
    existing_questions_str = json.dumps(existing_questions, indent=2) if existing_questions else "None"
    logger.debug(f"Generating batch: batch_size={batch_size}, random_seed={random_seed}, random_number={random_number}")
    logger.debug(f"Existing questions passed to LLM: {existing_questions_str}")

    system_prompt = (
        "You are an AI assistant specialized in generating {typeOfQuestions} questions. "
        "Using the provided context, follow these instructions to generate questions:\n\n"
        "1. **Question Requirements:**\n"
        "   - Create {batch_size} {typeOfQuestions} questions that are unique and diverse in their question text, answers, and options.\n"
        "   - Each question must have 4 options (A, B, C, D) if MCQ.\n"
        "   - Ensure the correct answers are present in the provided context.\n"
        "   - Avoid generating questions that match the following existing questions in terms of question text, answer, or options: {existing_questions_str}.\n"
        "   - Use this random seed to vary the questions: {random_seed}.\n"
        "   - Consider this number for additional variability: {random_number}.\n\n"
        "2. **Relevance Check:**\n"
        "   - If the chapter name '{chapterName}' or topic '{topicName}' is not explicitly mentioned in the context, respond with 'Out of syllabus'.\n\n"
        "3. **Output Format:**\n"
        "   - Provide the output in JSON format wrapped in ```json and ```.\n"
        "   - Each question and its options should be distinct properties within the JSON structure.\n"
        "   - Wrap the list of questions in a 'questions' key.\n\n"
        "Input Context:\n"
        "{context}"
    )
    logger.debug(f"System prompt: {system_prompt}")

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    response = await to_thread(rag_chain.invoke, {
        "input": f"Chapter: {chapterName}, Topic: {topicName}",
        "typeOfQuestions": typeOfQuestions,
        "batch_size": batch_size,
        "chapterName": chapterName,
        "topicName": topicName,
        "existing_questions_str": existing_questions_str,
        "random_seed": random_seed,
        "random_number": random_number
    })
    logger.debug(f"LLM response: {response['answer']}")

    if "Out of syllabus" in response["answer"]:
        logger.info("Chapter or topic not found in context")
        return {"message": "The requested chapter or topic is not covered in the provided context."}

    # Step 3: Parse generated questions
    try:
        json_str = extract_json_from_markdown(response["answer"])
        parsed_data = json.loads(json_str)
        if isinstance(parsed_data, dict) and "questions" in parsed_data:
            generated_questions = parsed_data["questions"]
        else:
            raise ValueError("Expected a dictionary with a 'questions' key containing a list")
        logger.info(f"Generated {len(generated_questions)} questions")
        logger.debug(f"Generated questions: {json.dumps(generated_questions, indent=2)}")
    except (ValueError, json.JSONDecodeError) as e:
        logger.error(f"Failed to parse generated questions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to parse generated questions: {str(e)}")

    # Step 4: Filter out duplicates with robust comparison
    seen = set()
    new_questions = []
    for q in generated_questions:
        q_tuple = (q["question"].lower().strip(), str(q.get("answer", "")).lower().strip(), tuple(q.get("options", {}).items()))
        if q_tuple not in seen and not any(question_equals(q, eq) for eq in existing_questions):
            seen.add(q_tuple)
            new_questions.append(q)
        else:
            logger.debug(f"Duplicate detected and skipped: {json.dumps(q, indent=2)}")
    logger.info(f"Filtered to {len(new_questions)} unique questions")
    logger.debug(f"New questions after filtering: {json.dumps(new_questions, indent=2)}")

    # Step 5: Select and shuffle questions
    random.shuffle(new_questions)
    questions_to_return = new_questions[:numberOfQuestions]
    logger.info(f"Returning {len(questions_to_return)} questions")

    # Step 6: Insert new questions into the database
    for question in questions_to_return:
        question_json = json.dumps(question)
        await db.execute(
            "INSERT INTO user_questions (user_id, chapter_name, topic_name, question) VALUES ($1, $2, $3, $4)",
            userId, chapterName, topicName, question_json
        )
    logger.info("New questions inserted into database")

    # Step 7: Return the unique questions
    return {"questions": questions_to_return}

# [Rest of the code remains unchanged: process_doubt, /ask endpoint, etc.]

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
    Give page number mention on the page from which response os taken
    Give output in JSON format with markdown as below:
    ```json
    {{
    "pageNumber": "{page_number}",
    "response": "Your answer here"
    }}```
    Context:
    "{context}"
    """
    
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content([system_prompt, f"User Query: {text}"])
    
    return {
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))

    # Working final code
