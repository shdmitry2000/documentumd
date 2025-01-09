import os
import pytest
from unittest.mock import patch, Mock, mock_open
from pathlib import Path
from app.tools.doc_proccesing.gemini_vertex_tool import GeminiVertexDocumentTool
from langchain_core.messages import AIMessage

class TestGeminiVertexDocumentTool:
    @pytest.fixture
    def tool(self):
        with patch('app.tools.doc_proccesing.gemini_vertex_tool.ChatVertexAI') as mock_chat:
            mock_chat._init_vertexai = Mock()
            tool = GeminiVertexDocumentTool()
            yield tool

    def test_initialization(self, tool):
        assert tool.settings["project"] == os.getenv("GOOGLE_CLOUD_PROJECT", "default-project")
        assert tool.settings["location"] == os.getenv("GOOGLE_CLOUD_LOCATION", "us-east5")

    @pytest.mark.parametrize("file_path,expected", [
        ("test.jpg", True),
        ("test.png", True),
        ("test.gif", True),
        ("test.webp", True),
        ("test.pdf", True),
        ("test.txt", False),
        ("test.doc", False),
    ])
    def test_is_supported_file_type(self, tool, file_path, expected):
        assert tool.is_supported_file_type(file_path) == expected

    def test_create_media_entry(self, tool):
        mime_type = "image/jpeg"
        data = "base64_encoded_data"
        entry = tool._create_media_entry(mime_type, data)
        
        assert entry["type"] == "media"
        assert entry["data"] == data
        assert entry["mime_type"] == mime_type

    @patch('builtins.open', new_callable=mock_open, read_data=b'test_data')
    def test_process_document(self, mock_file, tool):
        file_path = Path("test.jpg")
        
        # Mock processor methods
        tool.processor = Mock()
        tool.processor.get_mime_type.return_value = "image/jpeg"
        tool.processor.validate_mime_type.return_value = True
        
        result = tool._process_document(file_path)
        
        assert len(result) == 1
        assert result[0]["type"] == "media"
        assert result[0]["mime_type"] == "image/jpeg"
        assert "data" in result[0]

    def test_process_document_unsupported_type(self, tool):
        file_path = Path("test.txt")
        
        # Mock processor methods
        tool.processor = Mock()
        tool.processor.get_mime_type.return_value = "text/plain"
        tool.processor.validate_mime_type.return_value = False
        
        with pytest.raises(ValueError, match="Unsupported file type"):
            tool._process_document(file_path)

    @patch('app.tools.doc_proccesing.gemini_vertex_tool.ChatVertexAI')
    def test_analyze_documents(self, mock_chat, tool):
        # Mock file processing
        with patch.object(tool, '_process_document') as mock_process, \
             patch('pathlib.Path.exists', return_value=True):
            mock_process.return_value = [{"type": "media", "data": "test", "mime_type": "image/jpeg"}]
            
            # Mock ChatVertexAI response
            mock_instance = mock_chat.return_value
            mock_instance.invoke.return_value = AIMessage(content='{"test": "response"}')
            
            result = tool.analyze_documents(
                file_locations=["test.jpg"],
                system_prompt="Test prompt",
                question="Test question",
                return_json_only=True
            )
            
            assert mock_process.called
            assert mock_instance.invoke.called

    def test_analyze_documents_file_not_found(self, tool):
        with pytest.raises(FileNotFoundError):
            tool.analyze_documents(
                file_locations=["nonexistent.jpg"],
                system_prompt="Test prompt",
                question="Test question"
            )

    @patch('app.tools.doc_proccesing.gemini_vertex_tool.ChatVertexAI')
    def test_analyze_documents_multiple_files(self, mock_chat, tool):
        # Mock file processing
        with patch.object(tool, '_process_document') as mock_process, \
             patch('pathlib.Path.exists', return_value=True):
            mock_process.return_value = [{"type": "media", "data": "test", "mime_type": "image/jpeg"}]
            
            # Mock ChatVertexAI response
            mock_instance = mock_chat.return_value
            mock_instance.invoke.return_value = AIMessage(content='{"test": "response"}')
            
            result = tool.analyze_documents(
                file_locations=["test1.jpg", "test2.jpg"],
                system_prompt="Test prompt",
                question="Test question",
                return_json_only=True
            )
            
            assert mock_process.call_count == 2
            assert mock_instance.invoke.called
