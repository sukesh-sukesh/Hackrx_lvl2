



# main.py (Enhanced and Render-ready)
import os
import tempfile
import re
import json
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import requests
from fastapi import FastAPI, HTTPException, Depends, status, File, UploadFile, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Advanced RAG Document Processing System",
    description="LLM-powered system for processing natural language queries against structured documents",
    version="2.0.0",
    docs_url="/docs",  # Enable Swagger UI
    redoc_url="/redoc"  # Enable ReDoc
)

# Add CORS middleware for web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()
STATIC_TOKEN = "9409b1c8ad293eb74efa88c538211dcd22fb9eeaf89e7cefd1eb4b398f466c2f"

# Enhanced data models
class DecisionType(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    PENDING = "pending"
    REQUIRES_REVIEW = "requires_review"

@dataclass
class QueryContext:
    """Structured representation of parsed query"""
    age: Optional[int] = None
    gender: Optional[str] = None
    procedure: Optional[str] = None
    location: Optional[str] = None
    policy_duration: Optional[str] = None
    policy_type: Optional[str] = None
    raw_query: str = ""
    extracted_entities: Dict[str, Any] = None

class QueryRequest(BaseModel):
    """Basic query request model for compatibility"""
    documents: str  # URL to PDF
    questions: List[str]

class EnhancedQueryRequest(BaseModel):
    documents: str  # URL to PDF
    questions: List[str]
    context_extraction: bool = Field(default=True, description="Enable advanced context extraction")
    structured_response: bool = Field(default=True, description="Return structured decision response")

class ClauseReference(BaseModel):
    clause_id: Optional[str] = None
    clause_text: str
    relevance_score: float
    page_number: Optional[int] = None

class DecisionResponse(BaseModel):
    decision: DecisionType
    amount: Optional[float] = None
    currency: Optional[str] = "INR"
    justification: str
    referenced_clauses: List[ClauseReference]
    confidence_score: float
    extracted_context: Optional[Dict[str, Any]] = None

class QueryResponse(BaseModel):
    """Basic response model for compatibility"""
    answers: List[str]

class EnhancedQueryResponse(BaseModel):
    answers: List[str] = []
    structured_decisions: Optional[List[DecisionResponse]] = None
    processing_metadata: Dict[str, Any] = {}

# Authentication dependency
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != STATIC_TOKEN:
        logger.warning(f"Invalid token attempt: {credentials.credentials[:10]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

class QueryParser:
    """Advanced query parsing to extract structured information"""
    
    def __init__(self):
        self.patterns = {
            'age': [
                r'(\d+)\s*[yY](?:ear)?[sM]?(?:\s+old)?',  # 46Y, 46 years, 46M
                r'(?:age[:\s]+)?(\d+)',
                r'(\d+)\s*[Mm](?:ale)?',  # 46M, 46Male
                r'(\d+)\s*[Ff](?:emale)?'  # 46F, 46Female
            ],
            'gender': [
                r'(\d+)\s*([MF])',  # 46M, 46F
                r'\b([Mm]ale|[Ff]emale)\b',
                r'\b([Mm]|[Ff])\b(?!\d)'  # M or F not followed by digits
            ],
            'procedure': [
                r'\b([a-zA-Z\s]+(?:surgery|operation|procedure|treatment))\b',
                r'\b(knee|heart|brain|spine|dental|eye|cardiac)\s+\w+',
                r'\b(\w+(?:\s+\w+)*?)\s+surgery\b'
            ],
            'location': [
                r'\bin\s+([A-Za-z]+(?:\s+[A-Za-z]+)*)',
                r'\bat\s+([A-Za-z]+(?:\s+[A-Za-z]+)*)',
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*,?\s*(?:city|hospital)?'
            ],
            'policy_duration': [
                r'(\d+)\s*[-]?(?:month|mon|mo)\s*(?:old)?\s*(?:policy|insurance)',
                r'(\d+)\s*[-]?(?:year|yr)\s*(?:old)?\s*(?:policy|insurance)',
                r'(?:policy|insurance).*?(\d+)\s*(?:month|year)'
            ]
        }
    
    def parse_query(self, query: str) -> QueryContext:
        """Extract structured information from natural language query"""
        context = QueryContext(raw_query=query)
        extracted = {}
        
        # Extract age
        for pattern in self.patterns['age']:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                try:
                    context.age = int(match.group(1))
                    extracted['age'] = context.age
                    break
                except (ValueError, IndexError):
                    continue
        
        # Extract gender
        for pattern in self.patterns['gender']:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                gender_text = match.group(1) if len(match.groups()) == 1 else match.group(2)
                if gender_text.upper().startswith('M'):
                    context.gender = 'Male'
                elif gender_text.upper().startswith('F'):
                    context.gender = 'Female'
                extracted['gender'] = context.gender
                break
        
        # Extract procedure
        for pattern in self.patterns['procedure']:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                procedure = match.group(1).strip()
                if len(procedure) > 3:  # Filter out very short matches
                    context.procedure = procedure
                    extracted['procedure'] = context.procedure
                    break
        
        # Extract location
        for pattern in self.patterns['location']:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                location = match.group(1).strip()
                # Filter out common words that might be mistaken for locations
                if location.lower() not in ['the', 'and', 'with', 'for', 'surgery', 'old']:
                    context.location = location
                    extracted['location'] = context.location
                    break
        
        # Extract policy duration
        for pattern in self.patterns['policy_duration']:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                duration = match.group(1)
                context.policy_duration = f"{duration} months" if 'month' in match.group(0).lower() else f"{duration} years"
                extracted['policy_duration'] = context.policy_duration
                break
        
        context.extracted_entities = extracted
        return context

class EnhancedDocumentProcessor:
    """Enhanced document processing with clause identification"""
    
    def __init__(self):
        self.clause_patterns = [
            r'(?:clause|section|article)\s+(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\.\s*(?:[A-Z][a-z]+)',
            r'(?:condition|requirement|rule)\s+(\w+)'
        ]
    
    def process_documents_with_metadata(self, documents: List[Document]) -> List[Document]:
        """Process documents and add clause metadata"""
        enhanced_docs = []
        
        for doc in documents:
            # Try to identify clause numbers and sections
            content = doc.page_content
            clause_id = self._extract_clause_id(content)
            
            # Add metadata
            doc.metadata.update({
                'clause_id': clause_id,
                'content_type': self._classify_content_type(content),
                'importance_score': self._calculate_importance_score(content)
            })
            
            enhanced_docs.append(doc)
        
        return enhanced_docs
    
    def _extract_clause_id(self, content: str) -> Optional[str]:
        """Extract clause identifier from content"""
        for pattern in self.clause_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(1)
        return None
    
    def _classify_content_type(self, content: str) -> str:
        """Classify the type of content"""
        if any(word in content.lower() for word in ['exclusion', 'not covered', 'except']):
            return 'exclusion'
        elif any(word in content.lower() for word in ['coverage', 'covered', 'benefit']):
            return 'coverage'
        elif any(word in content.lower() for word in ['condition', 'requirement', 'must']):
            return 'condition'
        else:
            return 'general'
    
    def _calculate_importance_score(self, content: str) -> float:
        """Calculate importance score based on content"""
        important_words = ['coverage', 'benefit', 'exclusion', 'condition', 'requirement', 'amount', 'limit']
        score = sum(1 for word in important_words if word in content.lower())
        return min(score / len(important_words), 1.0)

def download_pdf(url: str) -> str:
    """Download PDF from URL and save to temporary file."""
    try:
        logger.info(f"Downloading PDF from URL: {url}")
        response = requests.get(url, timeout=30, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(response.content)
            tmp_file_path = tmp_file.name
            
        logger.info(f"PDF downloaded successfully to: {tmp_file_path}")
        return tmp_file_path
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download PDF: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to download PDF from URL: {str(e)}"
        )

def process_pdf_document(file_path: str):
    """Process PDF document and create enhanced vector store."""
    try:
        logger.info(f"Processing PDF document: {file_path}")
        
        # Load PDF document
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        if not documents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No content found in the PDF document"
            )
        
        logger.info(f"Loaded {len(documents)} pages from PDF")
        
        # Enhanced document processing
        doc_processor = EnhancedDocumentProcessor()
        enhanced_documents = doc_processor.process_documents_with_metadata(documents)
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        texts = text_splitter.split_documents(enhanced_documents)
        
        if not texts:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No text chunks generated from the PDF document"
            )
        
        logger.info(f"Created {len(texts)} text chunks")
        
        # Get Gemini API key
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="GEMINI_API_KEY environment variable not set"
            )
        
        # Create embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=gemini_api_key
        )
        
        # Create vector store
        vectorstore = FAISS.from_documents(texts, embeddings)
        
        logger.info("Enhanced vector store created successfully")
        return vectorstore, texts
        
    except Exception as e:
        logger.error(f"Error processing PDF document: {str(e)}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing PDF document: {str(e)}"
        )

def create_structured_decision_prompt(context: QueryContext) -> str:
    """Create a structured prompt for decision making"""
    
    context_str = ""
    if context.extracted_entities:
        context_str = f"""
EXTRACTED CONTEXT:
- Age: {context.age}
- Gender: {context.gender}
- Procedure: {context.procedure}
- Location: {context.location}
- Policy Duration: {context.policy_duration}
"""
    
    return f"""
You are an expert insurance claims processor. Analyze the provided policy documents and make a decision based on the query.

{context_str}

ORIGINAL QUERY: {context.raw_query}

Based on the policy documents provided, determine:
1. Whether the claim should be APPROVED, REJECTED, PENDING, or REQUIRES_REVIEW
2. The coverage amount (if applicable)
3. Specific clauses that support your decision
4. A clear justification

IMPORTANT: 
- Reference specific clauses, sections, or conditions from the documents
- Consider waiting periods, exclusions, coverage limits, and eligibility criteria
- Be precise about amounts and conditions
- If information is insufficient, mark as REQUIRES_REVIEW

Provide your response in the following JSON format:
{{
    "decision": "approved|rejected|pending|requires_review",
    "amount": <number_or_null>,
    "currency": "INR",
    "justification": "Clear explanation of the decision",
    "referenced_clauses": [
        {{
            "clause_text": "Exact text from the document",
            "relevance_score": 0.9,
            "page_number": 1
        }}
    ],
    "confidence_score": 0.8
}}
"""

def answer_questions_enhanced(
    vectorstore, 
    questions: List[str], 
    original_documents: List[Document],
    structured_response: bool = True
) -> EnhancedQueryResponse:
    """Enhanced question answering with structured decision making."""
    try:
        logger.info(f"Processing {len(questions)} questions with enhanced RAG")
        
        # Get Gemini API key
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="GEMINI_API_KEY environment variable not set"
            )
        
        # Initialize Gemini LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=gemini_api_key,
            temperature=0.1,  # Lower temperature for more consistent decisions
            convert_system_message_to_human=True
        )
        
        parser = QueryParser()
        answers = []
        structured_decisions = []
        processing_metadata = {}
        
        for i, question in enumerate(questions, 1):
            try:
                logger.info(f"Processing question {i}/{len(questions)}: {question[:50]}...")
                
                # Parse query for structured information
                query_context = parser.parse_query(question)
                processing_metadata[f'question_{i}_context'] = query_context.extracted_entities
                
                if structured_response:
                    # Create structured decision prompt
                    structured_prompt = create_structured_decision_prompt(query_context)
                    
                    # Create custom prompt template
                    prompt_template = PromptTemplate(
                        input_variables=["context", "question"],
                        template="""
Context from policy documents:
{context}

{question}

Please analyze the context and provide a structured decision.
"""
                    )
                    
                    # Create retrieval QA chain with custom prompt
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=vectorstore.as_retriever(
                            search_type="similarity",
                            search_kwargs={"k": 6}  # Retrieve more context for better decisions
                        ),
                        chain_type_kwargs={"prompt": prompt_template},
                        return_source_documents=True
                    )
                    
                    # Get structured response
                    result = qa_chain({"query": structured_prompt})
                    
                    # Try to parse JSON response
                    try:
                        if isinstance(result['result'], str):
                            # Extract JSON from response
                            json_match = re.search(r'\{.*\}', result['result'], re.DOTALL)
                            if json_match:
                                decision_data = json.loads(json_match.group())
                                
                                # Create structured decision response
                                structured_decision = DecisionResponse(
                                    decision=DecisionType(decision_data.get('decision', 'requires_review')),
                                    amount=decision_data.get('amount'),
                                    currency=decision_data.get('currency', 'INR'),
                                    justification=decision_data.get('justification', 'Analysis completed'),
                                    referenced_clauses=[
                                        ClauseReference(
                                            clause_text=clause['clause_text'],
                                            relevance_score=clause.get('relevance_score', 0.8),
                                            page_number=clause.get('page_number')
                                        ) for clause in decision_data.get('referenced_clauses', [])
                                    ],
                                    confidence_score=decision_data.get('confidence_score', 0.7),
                                    extracted_context=query_context.extracted_entities
                                )
                                
                                structured_decisions.append(structured_decision)
                                answers.append(structured_decision.justification)
                            else:
                                # Fallback to regular response
                                answers.append(result['result'])
                        else:
                            answers.append(str(result['result']))
                            
                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        logger.warning(f"Could not parse structured response for question {i}: {str(e)}")
                        answers.append(result['result'])
                
                else:
                    # Regular QA without structured response
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=vectorstore.as_retriever(
                            search_type="similarity",
                            search_kwargs={"k": 4}
                        ),
                        return_source_documents=False
                    )
                    
                    result = qa_chain.run(question)
                    answers.append(result)
                
                logger.info(f"Question {i} processed successfully")
                
            except Exception as e:
                logger.error(f"Error processing question {i}: {str(e)}")
                answers.append(f"Error processing question: {str(e)}")
        
        return EnhancedQueryResponse(
            answers=answers,
            structured_decisions=structured_decisions if structured_response else None,
            processing_metadata=processing_metadata
        )
        
    except Exception as e:
        logger.error(f"Error in enhanced question answering: {str(e)}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in enhanced question answering: {str(e)}"
        )

# MAIN ENDPOINT - Supports both basic and enhanced requests
@app.post("/hackrx/run")
async def run_rag_query(
    request: dict,  # Accept generic dict to handle both request types
    token: str = Depends(verify_token)
):
    """
    Main RAG endpoint - automatically detects request type and processes accordingly.
    Supports both basic QueryRequest and enhanced EnhancedQueryRequest formats.
    """
    temp_file_path = None
    
    try:
        logger.info("Received RAG query request")
        
        # Parse request - support both basic and enhanced formats
        documents = request.get("documents")
        questions = request.get("questions", [])
        context_extraction = request.get("context_extraction", True)
        structured_response = request.get("structured_response", True)
        
        # Validate input
        if not documents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Document URL is required"
            )
        
        if not questions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one question is required"
            )
        
        # Download PDF
        temp_file_path = download_pdf(documents)
        
        # Process PDF and create enhanced vector store
        vectorstore, original_documents = process_pdf_document(temp_file_path)
        
        # Answer questions with enhanced processing
        response = answer_questions_enhanced(
            vectorstore, 
            questions, 
            original_documents,
            structured_response=structured_response
        )
        
        logger.info("RAG query completed successfully")
        
        # Return appropriate response format based on structured_response flag
        if structured_response:
            return response.dict()
        else:
            # Return basic format for compatibility
            return {"answers": response.answers}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in RAG query: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.info(f"Cleaned up temporary file: {temp_file_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file {temp_file_path}: {str(e)}")

@app.post("/hackrx/upload")
async def run_rag_upload(
    pdf_file: UploadFile = File(...),
    questions: str = Form(...),
    token: str = Depends(verify_token)
):
    """
    Alternative endpoint for RAG-based question answering with direct PDF file upload.
    """
    temp_file_path = None
    
    try:
        logger.info("Received RAG upload request")
        
        # Validate file type
        if not pdf_file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only PDF files are supported"
            )
        
        # Parse questions from form data
        try:
            questions_list = json.loads(questions)
            if not isinstance(questions_list, list) or not questions_list:
                raise ValueError("Questions must be a non-empty array")
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid questions format. Expected JSON array: {str(e)}"
            )
        
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            content = await pdf_file.read()
            tmp_file.write(content)
            temp_file_path = tmp_file.name
            
        logger.info(f"PDF uploaded successfully to: {temp_file_path}")
        
        # Process PDF and create vector store
        vectorstore, original_documents = process_pdf_document(temp_file_path)
        
        # Answer questions with enhanced processing
        response = answer_questions_enhanced(
            vectorstore, 
            questions_list, 
            original_documents,
            structured_response=True
        )
        
        logger.info("RAG upload query completed successfully")
        return response.dict()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in RAG upload: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.info(f"Cleaned up temporary file: {temp_file_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file {temp_file_path}: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy", 
        "service": "Advanced RAG Document Processing System",
        "version": "2.0.0",
        "timestamp": str(pd.Timestamp.now()) if 'pd' in globals() else "2024"
    }

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Advanced RAG Document Processing System",
        "version": "2.0.0",
        "features": [
            "Advanced query parsing and context extraction",
            "Structured decision making with clause referencing",
            "Enhanced document processing with metadata",
            "Support for insurance claims processing",
            "Compatible with hackathon submission requirements"
        ],
        "endpoints": {
            "main_endpoint": "/hackrx/run",
            "upload_endpoint": "/hackrx/upload",
            "health_check": "/health",
            "docs": "/docs"
        },
        "authentication": "Bearer token required",
        "ready_for_submission": True
    }

# For Render deployment
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)

