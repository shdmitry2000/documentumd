import json
import mimetypes
import os
import base64
from pathlib import Path
import re
import sys
import traceback
from typing import Dict, Any, List, Optional, Union
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_vertexai.model_garden import ChatAnthropicVertex



sys.path.append(os.path.dirname('.'))
from app.tools.doc_proccesing.base_llm_tool import BaseLLMDocumentTool
import logging

logger = logging.getLogger(__name__)

class ClaudeVertexDocumentTool(BaseLLMDocumentTool):
    """Anthropic Claude implementation."""
    
    
    def __init__(self):
        super().__init__()
        self.settings = {    
            "location": os.getenv("GOOGLE_CLOUDE_LOCATION", "us-east5"),
            "project": os.getenv("GOOGLE_CLOUDE_PROJECT_ID","me-sb-dgcp-dpoc-pocyosh-pr"),
            "model": os.getenv("ANTHROPIC_LLM_MODEL_NAME", "claude-3-5-sonnet-v2@20241022")
        }
        
    @property
    def processor_name(self) -> str:
     
        return "claude_llm"
    
        
    def get_supported_mime_types(self) -> List[str]:
        return ["image/jpeg", "image/png", "image/gif", "image/webp", "application/pdf", "text/plain","application/json"]
    
    
    def _create_media_entry(self, mime_type: str, data: str) -> Dict[str, Any]:
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": mime_type,
                "data": data
            }
        }
    def _create_text_entry(self, text: str) -> Dict[str, Any]:
        return {"type": "text", "text": text}


    def _process_document(self, file_path: Path) -> List[Dict[str, Any]]:
        mime_type = self.processor.get_mime_type(str(file_path))
        
        # if not self.is_supported_mime_type(mime_type):
        #     raise ValueError(f"Unsupported file type: {mime_type}")


        if mime_type == "application/pdf":
            images = self.processor.pdf_to_base64_images(str(file_path))
            return [self._create_media_entry("image/jpeg", img) for img in images]
        elif mime_type.startswith("image/"):
            with open(file_path, "rb") as f:
                base64_data = base64.b64encode(f.read()).decode("utf-8")
            return [self._create_media_entry(mime_type, base64_data)]
        elif mime_type == "text/plain" or mime_type == "application/json":
           with open(file_path, "r") as f:
                text_data = f.read()
           return [self._create_text_entry(text_data)]
        else:
             raise ValueError(f"Unsupported file type: {mime_type}")
    
    
    @staticmethod
    def extract_json_from_response(response: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON from a text response that might contain additional content or markdown.
        Handles Unicode characters and normalizes quote characters.
        
        Args:
            response: String that may contain JSON, potentially in markdown code blocks
            
        Returns:
            Parsed JSON dict if found, None if no valid JSON found
        """    
        def clean_json_string(json_str: str) -> str:
            """Clean and normalize JSON string for parsing."""
            try:
                # Step 1: Basic string cleanup
               
                
                # Step 2: Handle escaped characters
                # First unescape any already escaped characters
                # cleaned = cleaned.encode().decode('unicode-escape')
                
                # Step 3: Fix quote issues by replacing all single and double quotes with double quotes
                cleaned = json_str.strip().replace("\'", '\u0027')
                cleaned = cleaned.replace("\"", '\u0022') 
                
                # Step 4: Properly escape quotes within string values
                # This uses a regex lookahead/lookbehind to only match quotes inside string values
                cleaned = re.sub(r'(?<!\\)"([^"]*)"(?=[,}\]])', r'"\1"', cleaned)
                
                # Step 5: Fix escaped backslashes
                cleaned = cleaned.replace('\\\\', '\\')
                
                # Step 6: Handle Hebrew characters
                # cleaned = re.sub(r'\\u([0-9a-fA-F]{4})', lambda m: chr(int(m.group(1), 16)), cleaned)

                return cleaned
                
            except Exception as e:
                logger.error(f"Error cleaning JSON string: {str(e)}")
                raise
            
            
        try:
            if isinstance(response, dict):
                return response
                
            # First try to parse the entire response as JSON
            try:
                # print("response",type(response),response)
                cleaned_response = clean_json_string(response)
                # print("cleaned_response",type(cleaned_response),cleaned_response)
                cleaned_response = json.loads(cleaned_response)
                # print("cleaned_response_after_load",type(cleaned_response),cleaned_response)
                return cleaned_response
            except json.JSONDecodeError:
                logger.error(f"Error extracting JSON: {traceback.format_exc()}")
                pass

            # Look for JSON in code blocks
            code_block_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
            code_blocks = re.findall(code_block_pattern, response)
            
            for block in code_blocks:
                try:
                    cleaned_block = clean_json_string(block.strip())
                    return json.loads(cleaned_block)
                except json.JSONDecodeError:
                    logger.error(f"Error extracting JSON: {traceback.format_exc()}")
                    continue
            
            # Try to find JSON pattern
            json_pattern = r"\{(?:[^{}]|(?R))*\}"
            matches = re.findall(json_pattern, response, re.DOTALL)
            
            for match in matches:
                try:
                    cleaned_match = clean_json_string(match.strip())
                    return json.loads(cleaned_match)
                except json.JSONDecodeError:
                    logger.error(f"Error extracting JSON: {traceback.format_exc()}")
                    continue
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting JSON: {str(e)}")
            logger.debug(f"Problematic response: {response}")
            return None
    
        
        
    def analyze_documents(self, 
                         file_locations: List[str],
                         system_prompt: str = "You are a document analysis expert.",
                         question: str = "Analyze this document.",
                         return_json_only: bool = False,
                         **kwargs) -> Union[str, Dict[str, Any]]:
        
        llm = ChatAnthropicVertex(
            model_name=self.settings["model"],
            location=self.settings["location"],
            project_id=self.settings["project"],
            max_tokens=8192,  # Increase max tokens
            temperature=0  # Set to 0 for consistent responses
        )
        
        data = []
        for file_location in file_locations:
            file_path = Path(file_location)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_location}")
            
            data.extend(self._process_document(file_path))
        
        
        if return_json_only:
            system_prompt += """ 
            You must provide complete responses without truncation.
            Generate the entire JSON structure in a single response.
            Do not split or abbreviate the output.
            Ensure all arrays and objects are complete and properly closed.
            """
            
            question += " Return complete JSON in a single response. Do not truncate or split the response."
            
        data.append({"type": "text", "text": question})
        
        results = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(data)
            ]
        )
        
        try:
            if return_json_only:
                outresult=results.content
                
                logger.debug(f"Raw response content: {outresult[:500]}...")  # Log first 500 chars
                
                json_data = self.extract_json_from_response(outresult)
                # print("json_data",type(json_data),json_data)
                if json_data:
                    return json_data
                logger.warning("No valid JSON found in response")
                return None
            
            return results.content
            
        except Exception as e:
            logger.error(f"Error combining JSON chunks: {str(e)} %s ",outresult[:500])
            traceback.print_exc()
            raise
            
            

if __name__ == "__main__":
   
    load_dotenv()

    # Choose your tool
    tool = ClaudeVertexDocumentTool()  # or ClaudeDocumentTool() or GPTDocumentTool()
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
        file_locations=[file_path1],
        system_prompt="You are a document analysis expert.",
        question="What is written here?",
        return_json_only=False
    )

    print(result)
