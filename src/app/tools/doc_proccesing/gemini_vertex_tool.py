import mimetypes
import os
import base64
from pathlib import Path
import sys
from typing import Dict, Any, List, Union
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_vertexai import ChatVertexAI
from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import logging

sys.path.append(os.path.dirname('.'))

from app.tools.doc_proccesing.base_llm_tool import BaseLLMDocumentTool

logger = logging.getLogger(__name__)




class GeminiVertexDocumentTool(BaseLLMDocumentTool):
    """Google Gemini implementation."""
    
    def __init__(self):
        super().__init__()
        self.settings = {
            "project": os.getenv("GOOGLE_CLOUD_PROJECT", "default-project"),
            "location": os.getenv("GOOGLE_CLOUD_LOCATION", "us-east5")
        }
        ChatVertexAI._init_vertexai(self.settings)
    
    @property
    def processor_name(self) -> str:
     
        return "gemini_llm"
        
    def get_supported_mime_types(self) -> List[str]:
        return ["image/jpeg", "image/png", "image/gif", "image/webp", "application/pdf","text/plain","application/json"]
    
    
      
    def _create_media_entry(self, mime_type: str, data: str) -> Dict[str, Any]:
        return {
            "type": "media",
            "data": data,
            "mime_type": mime_type
        }
        
    def _create_text_entry(self, text: str) -> Dict[str, Any]:
          return {"type": "text", "text": text}
       
            
    def _process_document(self, file_path: Path) -> List[Dict[str, Any]]:
        mime_type = self.processor.get_mime_type(str(file_path))
        
        if mime_type == "text/plain" or mime_type == "application/json":
            with open(file_path, "r") as f:
                text_data = f.read()
            return [self._create_text_entry(text_data)]
        else:
            
            with open(file_path, "rb") as f:
                base64_data = base64.b64encode(f.read()).decode("utf-8")
                
            if mime_type == "application/pdf":
                return [self._create_media_entry("application/pdf", base64_data)]
            elif mime_type.startswith("image/"):
                return [self._create_media_entry(mime_type, base64_data)]
            else:
                raise ValueError(f"Unsupported file type: {mime_type}")
            
        
    def analyze_documents(self, 
                         file_locations: List[str],
                         system_prompt: str = "You are a document analysis expert.",
                         question: str = "Analyze this document.",
                         model_name: str = os.getenv("GEMINI_LLM_MODEL_NAME", "gemini-2.0-flash-exp"),
                         return_json_only: bool = False,
                         **kwargs) -> Union[str, Dict[str, Any]]:
        
        llm = ChatVertexAI(model_name=model_name)
        data = []
        
        for file_location in file_locations:
            file_path = Path(file_location)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_location}")
            
            data.extend(self._process_document(file_path))
        
        if return_json_only:
            question += " Must return result as JSON!"
        
        data.append(question)
        
        results = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(data)
            ]
        )
        
        logger.info("results recived %s " %results.content)
        # print("results recived %s " %results.content)
        if return_json_only:
            json_data = self.extract_json_from_response(results.content)
            if json_data:
                return json_data
            logger.warning("No valid JSON found in response")
        
        return results.content
    
    
if __name__ == "__main__": 
    # Load environment variables
    load_dotenv()
    
    prompt="""
        Analyze this invoice and extract information according to the following ontology structure. 
        Return the data in valid JSON format that exactly matches this structure:
        
        {
            "invoice_number": string,
            "invoice_date": string?,
            "due_date": string?,
            "po_number": string? /* Purchase Order (P.O.) number is a unique identifier issued to a purchase order form. 
            It specifies the items/services a company desires to obtain from a specific provider/supplier. 
            Both buyer and seller use this number as a reference throughout the transaction.
            Common examples include:
            - "מס' הזמנה:2023242526" (Purchase Order number:2023242526)
            - 'PO-2023-001' (Purchase Order number: PO-2023-001)
            - 'הזמנת רכש:212132456' (Purchase Order 212132456)
            - 'הזמנה 1122334455' (Purchase Order number: 1122334455)
            - 'הזמנה מספר 1122334455' (Purchase Order number: 1122334455)
            */,
            "terms": string? /* Payment terms specified on the invoice that indicate:
            1. When payment is due (e.g., "Net 30", "Due in 60 days")
            2. Any early payment discounts (e.g., "2/10 Net 30" means 2% discount if paid within 10 days)
            3. Payment schedules (e.g., "50% advance, 50% upon delivery")
            4. Late payment penalties
            Common formats include:
            - "Net X" (payment due in X days)
            - "X% Y Net Z" (Y% discount if paid in X days, full amount due in Z days)
            - "End of Month" (payment due at month end)
            - "Immediate Payment" (payment due upon receipt)
            - "50% advance" (partial payment required upfront) */,
            "supplier": {
            "supplier_name": string?,
            "supplier_tax_id": string? /* 
            Tax identification number of the supplier entity .
            This number tipicly is an 8-9 character string. 
            In Israel it is a 9 character long number.
            It is very important to extract this field correctly.
            Common formats include:
            - 511315368 .פ.ח (tax_id)
            - ע.מ.511315368  (tax_id)
            - ח״פ:511315368  (tax_id)
            - עוסק מורשה 511315368  (tax_id)
            - מספר חברה 511315368  (tax_id)
            */,
            "supplier_address": string?,
            "supplier_website": string?,
            "supplier_email": string? /* 
            Email of the supplier. Example: info@acmecorp.com. Must include '@' symbol. 
            If no valid email is found, set this field to null.
            */,
            "supplier_phone": [string]?,
            "supplier_fax": string?
            },
            "customer": {
                "receiver_name": string,
                "receiver_tax_id": string? /* 
                Tax identification number of the receiving entity (customer). 
                This number is typically found near the customer's name or address, and is an 8-9 character string. 
                In Israel it is a 9 character long number. 
                Common formats include:
                - 520007030 .פ.ח (tax_id)
                - ע.מ.520007030  (tax_id)
                - ח״פ:520007030  (tax_id)
                - עוסק מורשה 520007030  (tax_id)
                - מספר חברה 520007030  (tax_id)
                
                */,
                "receiver_address": string?
            },
            "amounts": {
                    "subtotal": number,
                    "tax_rate": number?,
                    "tax_total": number?,
                    "total_amount": number
            },
            "payment_instructions": [{
                    "bank_name": string? /* 
                    Name of the bank.
                    */,
                    "bank_number": string? /* 
                    code of the bank .
                    */,
                    "branch_number": string? /* Branch number of the bank */,
                    "account_number": string? /* Account number */,
                    "account_name": string? /* Name of the account holder or company name */,
                    "iban": string? /* International Bank Account Number (IBAN) */,
                    "swift_code": string? /* Swift code for the bank */
            }]? /* 
                    A list of bank details for payment instructions.
                    Common formats include:
                    חשבון לזיכוי
                    העברה בנקאית
                    פרטים לעברה
                    לעברה בנקאית
                    לתשלום
                    פרטים לתשלום
                    ...
                    */,
            "notes": string? /* Notes field on the invoice that may contain:
            1. Special Payment Instructions
                - Tax exemption notices
                - Payment method requirements
                - Bank transfer instructions
                - Required payment reference numbers
                
            2. Delivery Information
                - Special delivery instructions
                - Delivery time restrictions
                - Contact person details
                
            3. Legal Notices
                - Tax-related declarations
                - Terms and conditions references
                - Copyright notices
                - Digital signature validations
                
            4. Service/Product Details
                - Service period specifications
                - Product warranty information
                - Usage restrictions
                - License terms
                
            5. Additional References
                - Purchase order references
                - Contract numbers
                - Project codes
                - Cost center information
                
            6. Discount/Promotion Information
                - Special offer details
                - Discount conditions
                - Loyalty program references */,
            "payment_details": [{
                    "payment_method": string?,
                    "payment_date": string?,
                    "payed_amount": number?,
                    "card_type": string?,
                    "card_number_last_digits": string?,
                    "bank": string?,
                    "bank_account_number": string?,
                    "iban": string?,
                    "swift_code": string?,
                    "check_number": string?
            }],
            "line_items": [{
                    "description": string?,
                    "quantity": number?,
                    "unit_price": number?,
                    "unit_price_before_tax": number?,
                    "amount": number?,
                    "tax_rate": number?,
                    "tax_amount": number?,
                    "product_code": string?
            }],
            "signature_assessment": {
            "has_signature": boolean /* Indicates if the document contains a signature */,
            "signature_type": string? /* Type of signature (e.g., 'handwritten', 'digital', 'stamp', 'qr_code') */,
            "signature_location": string? /* Location of signature in the document (e.g., 'bottom', 'top', 'margin') */,
            "verification_confidence": <class 'float'>? /* Confidence score for signature detection (0-100) */
        }
    }
        TECHNICAL ANALYSIS REQUIREMENTS:
            1. Readability Metrics:
            - Classify document type (text/image)
            - If image: 
                * OCR accuracy percentage
                * Text recognition confidence level
                * Detected image quality score
                
            2. Extraction Statistics:
            - Total fields processed
            - Extraction confidence score (0-100%)
            - Structured vs unstructured data ratio
            - Fields with highest/lowest confidence
            
            3. Processing Insights:
            - Complexity analysis of invoice layout
            - Potential data extraction challenges
            - Recommended preprocessing steps
            
            MANDATORY INSTRUCTIONS:
            - Include technical analysis as an additional JSON key "technical_assessment"
            - Add this technical assessment directly to the final JSON output
            - Ensure technical assessment provides quantitative and qualitative insights
            
            Important Processing Rules:
                1. You are document proccessing system of bank Discount.Make attention most documents will addressed the bank or it's sub divisions. 
                2. All monetary values should be numbers without currency symbols
                3. Use None for missing optional fields
                4. Dates must be in YYYY-MM-DD format
                5.Telephon number must include area codes ,country codes and shuld be full asap and properly formatted . 
                fax regulary begin with :פקס  
                6. Ensure all numbers are properly formatted decimals
                7. The JSON must be valid and match the exact structure above
                8. Round all monetary values to 2 decimal places
                9. If no country in invoice and you shure what is country according to address - you can add it.
                10. If no zipcode in invoice and you shure what is zipcode according to address - you can add it.
                11. If no currancy in invoice and you can guess it according to seller address.
                12.Email vs. Website (All Email Fields): 
                    The email fields must contain a valid email address, which includes the "@" symbol. 
                    If no valid email is found, set the field to null. 
                13.Automaticly check if document is signed and how and add the indication to metrics.
                14. use follow table to translate bank name to bank code in israel :
                    ID      Bank Name       Bank Name in english
                    1       ישראכרט בע"מ    Isracard Ltd
                    2       כרטיסי אשראי לישראל בע"מ        Israel Credit Cards Ltd
                    3       בנק אש ישראל בע״מ       Bank Esh Israel LTD
                    4       בנק יהב לעובדי המדינה בע"מ      Bank Yahav for State Employees Ltd
                    5       טרנזילה בע"מ    Tranzila Ltd
                    6       מקס איט פיננסים בע"מ    Max It Finance Ltd
                    7       קארדקום סליקה בע"מ      Cardcom Acquiring LTD
                    8       בנק הספנות לישראל בע"מ  Israel Shipping Bank Ltd
                    9       חברת בנק הדואר בע"מ     Postal Bank Company Ltd
                    10      בנק לאומי לישראל בע"מ   Bank Leumi Le-Israel Ltd
                    11      בנק דיסקונט לישראל בע"מ Israel Discount Bank Ltd
                    12      בנק הפועלים בע"מ        Bank Hapoalim Ltd
                    13      בנק אגוד לישראל בע"מ    Union Bank of Israel Ltd
                    14      בנק אוצר החייל בע"מ     Bank Otsar Hahayal Ltd
                    15      אופק אגודת אשראי שיתופית בע"מ   Ofek Credit Union Ltd
                    17      בנק מרכנתיל דיסקונט בע"מ        Mercantile Discount Bank Ltd
                    18      וואן זירו הבנק הדיגיטלי בע"מ    One Zero Digital Bank ltd
                    20      בנק מזרחי טפחות בע"מ    Bank Mizrahi-Tefahot Ltd
                    21      נימה שפע ישראל בע"מ     Neema Shefa Israel ltd
                    22      סיטיבאנק, אן.איי        Citibank N.A
                    23      אייצ' אס בי סי בנק      HSBC Bank plc
                    24      בנק אמריקאי ישראלי בע"מ American Israel Bank Ltd
                    25      בי אן פי פאריבס אס איי          BNP Paribas Israel
                    26      יובנק בע"מ      U-Bank Ltd
                    28      בנק קונטיננטל לישראל בע"מ       Continental Bank of Israel Ltd
                    31      הבנק הבינלאומי  או הבנק הבינלאומי הראשון לישראל בע"מ    First International Bank of Israel Ltd
                    31      הבנק בינלאומי    International Bank of Israel Ltd
                    32      בנק למימון ולסחר בע"מ   Finance and Trade Bank Ltd
                    33      בנק מרכנתיל לישראל בע"מ Mercantile Bank of Israel Ltd
                    34      בנק ערבי ישראלי בע"מ    Arab-Israeli Bank Ltd
                    35      גרואו פיימנטס בע"מ      GROW PAYMENTS LTD
                    37      בנק אלאורדון    Bank of Jordan
                    38      בנק אל תיג'ארי אלפלסטיני        Commercial Bank of Palestine
                    39      דה סטייט בנק אוף אינדיה         State Bank of India
                    43      בנק אלאהאלי אלאורדוני   Jordan National Bank
                    46      בנק מסד בע"מ    Bank Massad Ltd
                    47      גלובל רמיט שירותי מטבע בע"מ     Global Remit - Currency Services Ltd
                    48      קופת העובד הלאומי לאשראי וחיסכון נתניה  National Worker's Credit and Savings Fund Netanya
                    49      אלבנק אלערבי    Arab Bank plc
                    50      מרכז סליקה בנקאי בע"מ   Bank Settlement Center Ltd. (MASAV)
                    52      בנק פועלי אגודת ישראל בע"מ      Poalei Agudat Yisrael Bank Ltd
                    54      בנק ירושלים בע"מ        Bank of Jerusalem Ltd
                    58      רי ווייר א ס ג מחקר ופיתוח בע"מ REWIRE (O.S.G) RESEARCH AND DEVELOPMENT LTD
                    59      שירותי בנק אוטומטיים    Automatic Bank Services (Shva)
                    62      שירותי קורספונדנציה בע"מ        Correspondent Services Ltd
                    66      בנק אלקאהירה עמאן       Cairo-Amman Bank
                    67      בנק אלעקארי אלערבי      Arab Land Bank
                    68      בנק מוניציפל בע"מ       Municipal Bank Ltd
                    69      גי אם טי טק אינוביישן בע"מ      GMT Tech Innovation LTD
                    71      בנק אלאורדון ואלחליג'   Jordan Gulf Bank
                    73      בנק אלאסלאמי אלערבי     Arab Islamic Bank
                    74      בנק HSBC המזרח התיכון   HSBC Bank Middle East
                    75      וי צ'ק בע"מ      V-CHECK LTD
                    76      בנק אלאסתתמאר אלפלסטיני         Palestine Investment Bank
                    78      רבולוט  REVOLUT LTD
                    79             019 שירותי תשלום בע"מ      019 Payment Services ltd
                    82      בנק אלקודס ללתמניה וללסתתמר     Al-Quds Bank for Development and Investment
                    83      בנק אלאתיחאד ללדיכאר ואלאסתתמאר Union Bank for Savings and Investment
                    84      בנק אלאסכאן     The Housing Bank
                    85      יופיי סליקה בע"מ        UPAY ACQUIRING LTD
                    86      אטמס מטריקס בע"מ        A.T.M.S Matrix Ltd
                    89      בנק פלסטין      Bank of Palestine

                15.**Global Address Handling Instruction:**

                    1.  **Multiple Address Detection:**
                    *   The system must be capable of recognizing when multiple addresses are present in the document.
                        This may be indicated by a clear visual separation (e.g., separate lines, different formatting),
                        the presence of connecting words ("and"), or multiple address-related keywords and structures.

                    2.  **Address Separation and Assignment:**
                        *   If multiple addresses are detected, the system will assign them based on the context of their appearance. 
                        The  actual assignment may vary based on the prompt's specific instructions.
                        
                            
                    3.  **Address Components:**
                        *   The system should extract all components of each address (the tower or plase name if exists, street, housenumber, city, postal code, country) if possible.
                        *   If certain parts of the address cannot be extracted, the system should include the available parts in the output string, maintaining the order (the tower or plase name if exists,street,housenumber, city, postal code, country), if applicable.

                    4. as proffesional clerk recheck all address correctness after extraction as additional step.
                
            Final Output: Complete JSON with integrated technical assessment 
    """

    # Choose your tool
    tool = GeminiVertexDocumentTool()  # or ClaudeDocumentTool() or GPTDocumentTool()
    print("begin")
    DATA_FOLDER_TEST="./src/documents_prompt_creating/tools/test_docs/"
    file_path=f"{DATA_FOLDER_TEST}InvoiceExample.pdf"

    file_path="/Users/dmitrysh/code/documentumd/src/invoices/invoice4u_1.png"
    file_path=f"/Users/dmitrysh/code/google_cloude/google_rag_test/app/modules/llm/llmtest/Invoice.pdf"

    # file_path1=f"/Users/dmitrysh/code/google_cloude/google_rag_test/invoices_copy/INVOICE.png"
    
    file_path_text=f"/Users/dmitrysh/code/documentumd/tests/ontology_creating/creating.prompt.txt"

    # Analyze documents
    result = tool.analyze_documents(
        file_locations=[file_path],
        system_prompt="Analyze this invoice according to the schema...",
        question="Extract invoice details",
        return_json_only=True
    )
    
    # result = tool.analyze_documents(
    #     file_locations=[file_path_text],
    #     system_prompt="You are a document analysis expert.",
    #     question="What is written here?",
    #     return_json_only=False
    # )

    print(result)
