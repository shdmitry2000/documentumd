from abc import ABC, abstractmethod
import json
import mimetypes
import os
from pathlib import Path
import re
import sys
from typing import Dict, Any, List, Optional, Union
from venv import logger
from dotenv import load_dotenv
from json import JSONDecoder

sys.path.append(os.path.dirname('.'))

from app.tools.doc_proccesing.document_processor import DocumentProcessor







class BaseDocumentTool(ABC):
    
    
    
    def proccess_cli(self,file_path,prompt,question="",return_json_only=True) :
        try:
            result = self.analyze_documents(
                file_locations=[file_path],
                system_prompt=prompt,
                question=question,
                return_json_only=return_json_only
            )
            return (file_path,result,None)
        except Exception as e:
                logger.error(f"Error proccess_cli : {str(e)}")
                return (file_path,None,e)
            
    @abstractmethod
    def is_supported_file_type(self,file_path: str) -> bool:
        pass 
    
    @property
    @abstractmethod
    def processor_name(self) -> str:
        pass
        
       
    
    @abstractmethod
    def analyze_documents(self,file_path: str) -> bool:
        pass 

class BaseLLMDocumentTool(BaseDocumentTool):
    """Base class for LLM document processing tools."""
    
    def __init__(self):
        load_dotenv()
        self.processor = DocumentProcessor()
  

    @abstractmethod
    def _create_media_entry(self, mime_type: str, data: str) -> Dict[str, Any]:
        """Create provider-specific media entry."""
        pass

    @abstractmethod
    def get_supported_mime_types(self) -> List[str]:
        """Create provider-specific media type supported."""
        pass

    def is_supported_mime_type(self,mime_type: str) -> bool:
        """Checks if the file type is supported ."""
        
        return mime_type in self.get_supported_mime_types() if mime_type else False 
    
    def is_supported_file_type(self,file_path: str) -> bool:
        """Checks if the file type is supported ."""

        mime_type, _ = mimetypes.guess_type(file_path)
        return self.is_supported_mime_type(mime_type)
      
      
    @abstractmethod
    def _process_document(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process a single document."""
        pass

    @abstractmethod
    def analyze_documents(self, 
                         file_locations: List[str],
                         system_prompt: str = "You are a document analysis expert.",
                         question: str = "Analyze this document.",
                         return_json_only: bool = False,
                         **kwargs) -> Union[str, Dict[str, Any]]:
        """Analyze documents using the LLM."""
        pass
    
    @abstractmethod
    def analyze_documents(self, 
                         file_locations: List[str],
                         system_prompt: str = "You are a document analysis expert.",
                         question: str = "Analyze this document.",
                         model_name: str = "gemini-1.5-flash-002",
                         return_json_only: bool = False,
                         **kwargs) -> Union[str, Dict[str, Any]]:
        pass
    
    @staticmethod
    def extract_json_objects(text, decoder=JSONDecoder()):
        """Find JSON objects in text, and yield the decoded JSON data

        Does not attempt to look for JSON arrays, text, or other JSON types outside
        of a parent JSON object.

        """
        pos = 0
        while True:
            match = text.find('{', pos)
            if match == -1:
                break
            try:
                # print("match",match)
                result, index = decoder.raw_decode(text[match:])
                # print(result, index)
                yield result
                pos = match + index
            except ValueError:
                pos = match + 1


    @staticmethod
    def extract_json_values(text, decoder=JSONDecoder()):
        """Find JSON values (objects, arrays, primitives) in text, and yield the decoded JSON data.

        This will attempt to decode any valid JSON value found, including
        objects, arrays, strings, numbers, booleans, and null.
        """
        pos = 0
        while True:
            try:
                # Attempt to decode starting from the current position
                result, index = decoder.raw_decode(text[pos:])
                yield result
                pos += index # Move to after the decoded part
            except ValueError as e:
                if "Expecting value" in str(e):
                    # if no more json value found, break
                    break
                pos += 1 # Move to next char and continue looking
            except Exception as e:
                #catch all errors for safer extraction
                pos+=1
                 
    @staticmethod
    def extract_json_from_response(response: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON from a text response that might contain additional content or markdown.
        
        Args:
            response: String that may contain JSON, potentially in markdown code blocks
            
        Returns:
            Parsed JSON dict if found, None if no valid JSON found
        """
        ex=None
        try:
            if type(response) is dict:
                return response
            # First try to parse the entire response as JSON
            return json.loads(response)
        except json.JSONDecodeError as e:
            try:
                ex=e
                # Remove markdown code blocks if present
                # Match content between ```json and ``` or just between ```
                code_block_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
                
                code_blocks = re.findall(code_block_pattern, response)
                json_list=[]
                # Try each code block
                for block in code_blocks:
                    try:
                        stripped_block=block.strip()
                        # Remove trailing commas in arrays and objects
                        sanitized_block = re.sub(r',(\s*[\]}])', r'\1', stripped_block)
    
    
                        # sanitized_block = stripped_block.replace(r"\"", ' ').replace(r"'", ' ').replace(r"\n", ' ').replace("\"", '\u0022').replace(r"\\'", "'")
                        # print("sanitized_block",sanitized_block)
                        # .replace("\'", '\u0027')
                        # .replace(r'\\"', '"').replace("\'", '\u0027').replace("\"", '\u0022') 

                
                        # .replace(r'\\'', ''')
                        # replace('\\r', '').replace('\\n','').replace('\\0','"')
                        # sanitized_block = stripped_block.replace("\\\"", '\u0027').replace("\\\'", '\u0022').replace('\'', '\u0027').replace('\"', '\u0022')
                        # sanitized_block = re.sub(r'(?<!\\)"([^"]*)"(?=[,}\]])', r'"\1"', stripped_block)


                
                        # sanitized_block=stripped_block
                        
                        return json.loads(sanitized_block)
                    except json.JSONDecodeError as e:
                        ex=e
                        
                       
                        # print(BaseLLMDocumentTool.extract_json_values(filtered_string))
                        # for json_object in BaseLLMDocumentTool.extract_json_objects(sanitized_block):
                        #     json_list.append(json_object)
                        # print(json_list)
                        # return json_list
                        # json_iterator=BaseLLMDocumentTool.extract_json_objects(sanitized_block)
                        # return next(json_iterator,None)
           
                        continue
                
                # If no code blocks or valid JSON in code blocks, try to find JSON pattern
                json_pattern = r"\{(?:[^{}]|(?R))*\}"
                matches = re.findall(json_pattern, response, re.DOTALL)
                
                for match in matches:
                    try:
                        return json.loads(match)
                    except json.JSONDecodeError as e:
                        ex=e
                    except json.JSONDecodeError as e:
                        ex=e    
                        continue
                if ex :
                    raise ex
                else:
                    return None
            except Exception as e:
                logger.error(f"Error extracting JSON: {str(e)}")
                return None


if __name__ == "__main__":
    teststr="""
    ```json\n{\n\t\"invoice_number\": \"SI216000257\",\n\t\"invoice_date\": \"2021-04-29\",\n\t\"due_date\": \"2021-05-31\",\n\t\"po_number\": null,\n\t\"terms\": null,\n\t\"supplier\": {\n\t\t\"supplier_name\": \"תפן (אי.אל) ישראל בע\\\"מ\",\n\t\t\"supplier_tax_id\": \"513340760\",\n\t\t\"supplier_address\": \"שד' מנחם בגין 5\\nבית דגן 50200\",\n\t\t\"supplier_website\": null,\n\t\t\"supplier_email\": null,\n\t\t\"supplier_phone\": [\n\t\t\t\"03-9775151\"\n\t\t],\n\t\t\"supplier_fax\": \"03-9775152\"\n\t},\n\t\"customer\": {\n\t\t\"receiver_name\": \"בנק דיסקונט לישראל בע\\\"מ\",\n\t\t\"receiver_tax_id\": \"520007030\",\n\t\t\"receiver_address\": \"רח' אצ\\\"ל 53,א.תעשיה חדש קומה 5\\nראשל\\\"צ\"\n\t},\n\t\"amounts\": {\n\t\t\"subtotal\": 70000.00,\n\t\t\"tax_rate\": 17.00,\n\t\t\"tax_total\": 11900.00,\n\t\t\"total_amount\": 81900.00\n\t},\n\t\"payment_instructions\": [\n\t\t{\n\t\t\t\"bank_name\": \"בנק בינלאומי\",\n\t\t\t\"bank_number\": \"31\",\n\t\t\t\"branch_number\": \"126\",\n\t\t\t\"account_number\": \"119792\",\n\t\t\t\"account_name\": null,\n\t\t\t\"iban\": null,\n\t\t\t\"swift_code\": null\n\t\t}\n\t],\n\t\"notes\": null,\n\t\"payment_details\": [],\n\t\"line_items\": [\n\t\t{\n\t\t\t\"description\": \"יישום יוזמות אסטרטגיות חטיבה עסקית מסחרית\\nחיוב בגין שעות מה - 20.04 עד 19.05\\nמוחמד מטר - 70 שעות\\nעידן שירזי - 100 שעות\\nאייל כהן - 100 שעות\",\n\t\t\t\"quantity\": 1.00,\n\t\t\t\"unit_price\": 70000.00,\n\t\t\t\"unit_price_before_tax\": null,\n\t\t\t\"amount\": 70000.00,\n\t\t\t\"tax_rate\": null,\n\t\t\t\"tax_amount\": null,\n\t\t\t\"product_code\": \"436\"\n\t\t}\n\t],\n\t\t\"signature_assessment\": {\n\t\t\"has_signature\": true,\n\t\t\"signature_type\": \"digital\",\n\t\t\"signature_location\": \"bottom\",\n\t\t\"verification_confidence\": null\n\t},\n\"technical_assessment\": {\n    \"readability_metrics\": {\n      \"document_type\": \"image\",\n      \"ocr_accuracy_percentage\": 98.0,\n      \"text_recognition_confidence_level\": \"high\",\n      \"detected_image_quality_score\": 0.95\n    },\n    \"extraction_statistics\": {\n      \"total_fields_processed\": 23,\n      \"extraction_confidence_score\": 97.0,\n      \"structured_vs_unstructured_data_ratio\": \"80:20\",\n\t\t\"fields_with_highest_confidence\":[\"supplier_name\", \"supplier_tax_id\", \"invoice_number\", \"invoice_date\", \"due_date\", \"subtotal\", \"total_amount\", \"tax_total\",\"bank_name\",\"bank_number\",\"branch_number\",\"account_number\",\"receiver_name\",\"receiver_tax_id\", ],\n\t\t\"fields_with_lowest_confidence\": [\"supplier_website\", \"supplier_email\", \"terms\",\"po_number\",\"payment_method\",\"card_type\",\"card_number_last_digits\",\"bank\",\"bank_account_number\",\"iban\",\"swift_code\",\"check_number\",\"payment_date\",\"payed_amount\",\"account_name\"]\n    },\n    \"processing_insights\": {\n      \"complexity_analysis_of_invoice_layout\": \"The invoice has a standard layout with clear sections for supplier, customer, and amounts. There is a clear distinction between the header, main body, and footer elements, simplifying the extraction process. The presence of a digital signature also helps in authenticity verification.\",\n      \"potential_data_extraction_challenges\": \"The presence of multiple addresses and phone numbers may require careful extraction logic to ensure correct assignment to supplier or customer. The use of Hebrew language might pose challenges for some general-purpose OCR tools, necessitating the use of language-specific models. Some fields like account name are not clearly provided so has to be extracted from different source if needed.\",\n      \"recommended_preprocessing_steps\": \"While the image quality is good, some preprocessing steps like deskewing and noise reduction could improve OCR accuracy further if needed. Also, the system should perform a check if the provided bank name match to a valid code, if not must extract from other source or create a new reference.\"\n    }\n  }\n}\n```
    """
    
    
    print(BaseLLMDocumentTool.extract_json_from_response(teststr))
    # for result in BaseLLMDocumentTool.extract_json_objects(teststr):
    #      print(result)