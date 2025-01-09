from crewai.tools import BaseTool
from typing import List, Type, Union
from pydantic import BaseModel, Field

from app.tools.doc_proccesing.claude_vertex_tool import ClaudeVertexDocumentTool
from app.tools.doc_proccesing.gemini_vertex_tool import GeminiVertexDocumentTool
from app.tools.doc_proccesing.gpt_tool import GPTDocumentTool

class AskDocPromptSchema(BaseModel):
    """Input for Vision Tool."""

    image_path_urls: Union[ List[str] ]= ["The image paths or URL."]
    system_prompt: str = "The system prompt. Can be empty"
    question_prompt: str = "the question about document. Can be empty"
    answer_as_json: bool = True

    


class CrewaiGeminiVertexDocumentProccessorTool(BaseTool):
    name: str = "Gemini_Vertex_Documents_proccessor"
    description: str = (
        "ask questions about documents.Take documents from local path , upload to gemini and get answers.One of system_prompt or question_prompt cant be empty and shold content question"
    )
    args_schema: Type[BaseModel] = AskDocPromptSchema

    def _run(self, **kwargs) -> str:
    
        image_path_urls = kwargs.get("image_path_urls")
        system_prompt = kwargs.get("system_prompt")
        question_prompt = kwargs.get("question_prompt")
        answer_as_json = kwargs.get("answer_as_json")


        if not image_path_urls or image_path_urls == [] or (not system_prompt and not question_prompt) :
            return "Image Path or URLs is required and prompts cant be empty."

        tool = GeminiVertexDocumentTool() 
        
        result = tool.analyze_documents(
        file_locations=image_path_urls,
        system_prompt=system_prompt,
        question=question_prompt,
        return_json_only=answer_as_json
        )
        
        return result
        
class CrewaiClaudeVertexDocumentProccessorTool(BaseTool):
    name: str = "Claude_Vertex_Documents_proccessor"
    description: str = (
        "ask questions about documents.Take documents from local path , upload to gemini and get answers.One of system_prompt or question_prompt cant be empty and shold content question"
    )
    args_schema: Type[BaseModel] = AskDocPromptSchema

    def _run(self, **kwargs) -> str:
    
        image_path_urls = kwargs.get("image_path_urls")
        system_prompt = kwargs.get("system_prompt")
        question_prompt = kwargs.get("question_prompt")
        answer_as_json = kwargs.get("answer_as_json")


        if not image_path_urls or image_path_urls == [] or (not system_prompt and not question_prompt) :
            return "Image Path or URLs is required and prompts cant be empty."

        tool = ClaudeVertexDocumentTool() 
        
        result = tool.analyze_documents(
        file_locations=image_path_urls,
        system_prompt=system_prompt,
        question=question_prompt,
        return_json_only=answer_as_json
        )
        
        return result
    
    
class CrewaiGPTDocumentProccessorTool(BaseTool):
    name: str = "GPT_Documents_proccessor"
    description: str = (
        "ask questions about documents.Take documents from local path , upload to gemini and get answers.One of system_prompt or question_prompt cant be empty and shold content question"
    )
    args_schema: Type[BaseModel] = AskDocPromptSchema

    def _run(self, **kwargs) -> str:
    
        image_path_urls = kwargs.get("image_path_urls")
        system_prompt = kwargs.get("system_prompt")
        question_prompt = kwargs.get("question_prompt")
        answer_as_json = kwargs.get("answer_as_json")


        if not image_path_urls or image_path_urls == [] or (not system_prompt and not question_prompt) :
            return "Image Path or URLs is required and prompts cant be empty."

        tool = GPTDocumentTool() 
        
        result = tool.analyze_documents(
        file_locations=image_path_urls,
        system_prompt=system_prompt,
        question=question_prompt,
        return_json_only=answer_as_json
        )
        
        return result