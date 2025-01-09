import mimetypes
import os
import base64
from pathlib import Path
import sys
from typing import Dict, Any, List, Union
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

import logging

sys.path.append(os.path.dirname('.'))

from app.tools.doc_proccesing.base_llm_tool import BaseLLMDocumentTool


 
logger = logging.getLogger(__name__)




logger = logging.getLogger(__name__)

class GPTDocumentTool(BaseLLMDocumentTool):
    """OpenAI GPT implementation."""
    
    
    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model_name=os.getenv("OPENAI_MODEL")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

    @property
    def processor_name(self) -> str:
        return "chatgpt_llm"
    
    def get_supported_mime_types(self) -> List[str]:
        return ["image/jpeg", "image/png", "image/gif", "image/webp", "application/pdf","text/plain","application/json"]
    
    
    def _create_media_entry(self, mime_type: str, data: str) -> Dict[str, Any]:
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{data}",
                "detail": "high"
            }
        }

    def _create_text_entry(self, text: str) -> Dict[str, Any]:
        return {"type": "text", "text": text}


    
        
    def _process_document(self, file_path: Path) -> List[Dict[str, Any]]:
        mime_type = self.processor.get_mime_type(str(file_path))
        
        if mime_type == "text/plain" or mime_type == "application/json":
            with open(file_path, "r") as f:
                text_data = f.read()
            return [self._create_text_entry(text_data)]
        
        elif mime_type == "application/pdf":
            images = self.processor.pdf_to_base64_images(str(file_path))
            return [self._create_media_entry("image/jpeg", img) for img in images]
        
        elif mime_type.startswith("image/"):
            with open(file_path, "rb") as f:
                base64_data = base64.b64encode(f.read()).decode("utf-8")
            return [self._create_media_entry(mime_type, base64_data)]
        else:
                raise ValueError(f"Unsupported file type: {mime_type}")
            


    def analyze_documents(self, 
                         file_locations: List[str],
                         system_prompt: str = "You are a document analysis expert.",
                         question: str = "Analyze this document.",
                         return_json_only: bool = False,
                         **kwargs) -> Union[str, Dict[str, Any]]:
        
        llm = ChatOpenAI(model=self.model_name)
        data = []
        
        for file_location in file_locations:
            file_path = Path(file_location)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_location}")
            
            data.extend(self._process_document(file_path))
        
        if return_json_only:
            question += " Must return result as JSON!"
        
        data.append({"type": "text", "text": question})
        
        results = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(data)
            ]
        )
        
        if return_json_only:
            json_data = self.extract_json_from_response(results.content)
            if json_data:
                return json_data
            logger.warning("No valid JSON found in response")
        
        return results.content
    
    
  

if __name__ == "__main__":
   
    load_dotenv()

    llm = ChatOpenAI(model="gpt-4o")
    # Choose your tool
    tool = GPTDocumentTool()  # or ClaudeDocumentTool() or GPTDocumentTool()
    print("begin")
    DATA_FOLDER_TEST="./src/documents_prompt_creating/tools/test_docs/"
    file_path=f"{DATA_FOLDER_TEST}InvoiceExample.pdf"

    file_path=f"/Users/dmitrysh/code/google_cloude/google_rag_test/app/modules/llm/llmtest/Invoice.pdf"

    file_path1=f"/Users/dmitrysh/code/google_cloude/google_rag_test/invoices_copy/INVOICE.png"
    
    file_path_text=f"/Users/dmitrysh/code/documentumd/tests/ontology_creating/creating.prompt.txt"

    # Analyze documents
    # result = tool.analyze_documents(
    #     file_locations=[file_path],
    #     system_prompt="Analyze this invoice according to the schema...",
    #     question="Extract invoice details",
    #     return_json_only=True
    # )
    
    result = tool.analyze_documents(
        file_locations=[file_path_text],
        system_prompt="You are a document analysis expert.",
        question="What is written here?",
        return_json_only=False
    )

    print(result)

