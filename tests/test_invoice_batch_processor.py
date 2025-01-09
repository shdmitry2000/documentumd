import os
import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from datetime import datetime
from decimal import Decimal

from src.invoice_batch_proccessor import BatchProcessor, DecimalEncoder
from app.tools.doc_proccesing.base_llm_tool import BaseDocumentTool
from invoice_ontology import InvoiceOntology

class MockTool(BaseDocumentTool):
    def is_supported_file_type(self, file_path: str) -> bool:
        return True
        
    def proccess_cli(self, file_path: str, prompt: str, return_json_only: bool = False):
        return True, {"test": "result"}, None

@pytest.fixture
def mock_dirs(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    error_dir = tmp_path / "error"
    
    input_dir.mkdir()
    output_dir.mkdir()
    error_dir.mkdir()
    
    return input_dir, output_dir, error_dir

@pytest.fixture
def processor(mock_dirs):
    input_dir, output_dir, error_dir = mock_dirs
    return BatchProcessor(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        error_dir=str(error_dir),
        tool=MockTool(),
        default_ontology_class=InvoiceOntology,
        validation=False
    )

class TestDecimalEncoder:
    def test_decimal_encoding(self):
        encoder = DecimalEncoder()
        assert encoder.default(Decimal('10.5')) == 10.5
        
        with pytest.raises(TypeError):
            encoder.default("not a decimal")

class TestBatchProcessor:
    def test_initialization(self, mock_dirs):
        input_dir, output_dir, error_dir = mock_dirs
        processor = BatchProcessor(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            error_dir=str(error_dir),
            tool=MockTool(),
            default_ontology_class=InvoiceOntology
        )
        
        assert processor.input_dir == input_dir
        assert processor.output_dir == output_dir
        assert processor.error_dir == error_dir
        assert isinstance(processor.tool, MockTool)
        assert processor.default_ontology_class == InvoiceOntology
        assert processor.validation is False
        
        assert output_dir.exists()
        assert error_dir.exists()

    def test_load_custom_prompt(self, processor, mock_dirs):
        input_dir = mock_dirs[0]
        prompt_file = input_dir / "test.prompt"
        test_prompt = "Test prompt content"
        
        with open(prompt_file, 'w') as f:
            f.write(test_prompt)
            
        loaded_prompt = processor._load_custom_prompt(prompt_file)
        assert loaded_prompt == test_prompt

    def test_load_custom_prompt_error(self, processor):
        nonexistent_file = Path("nonexistent.prompt")
        loaded_prompt = processor._load_custom_prompt(nonexistent_file)
        assert loaded_prompt is None

    def test_load_custom_ontology(self, processor, mock_dirs):
        input_dir = mock_dirs[0]
        ontology_file = input_dir / "test_ontology.py"
        
        ontology_content = """
from invoice_ontology import InvoiceOntology

class CustomInvoiceOntology(InvoiceOntology):
    pass
"""
        with open(ontology_file, 'w') as f:
            f.write(ontology_content)
            
        loaded_ontology = processor._load_custom_ontology(ontology_file)
        assert loaded_ontology is not None
        assert issubclass(loaded_ontology, InvoiceOntology)

    def test_load_custom_ontology_error(self, processor):
        nonexistent_file = Path("nonexistent_ontology.py")
        loaded_ontology = processor._load_custom_ontology(nonexistent_file)
        assert loaded_ontology is None

    def test_get_processing_config_default(self, processor, mock_dirs):
        input_dir = mock_dirs[0]
        test_file = input_dir / "test.pdf"
        test_file.touch()
        
        prompt, ontology = processor.get_processing_config(test_file)
        assert isinstance(prompt, str)
        assert ontology == InvoiceOntology

    def test_get_processing_config_custom_prompt(self, processor, mock_dirs):
        input_dir = mock_dirs[0]
        test_file = input_dir / "test.pdf"
        test_file.touch()
        
        prompt_file = input_dir / "test.prompt"
        test_prompt = "Custom test prompt"
        with open(prompt_file, 'w') as f:
            f.write(test_prompt)
            
        prompt, ontology = processor.get_processing_config(test_file)
        assert prompt == test_prompt
        assert ontology == InvoiceOntology

    def test_process_file_success(self, processor, mock_dirs):
        input_dir = mock_dirs[0]
        test_file = input_dir / "test.pdf"
        test_file.touch()
        
        success, error = processor.process_file(test_file)
        assert success is True
        assert error is None
        
        # Check output file
        output_file = processor.output_dir / "test_processed.json"
        assert output_file.exists()
        
        with open(output_file) as f:
            result = json.load(f)
            assert "test" in result
            assert "metadata" in result
            assert result["metadata"]["source_file"] == str(test_file)

    def test_process_file_unsupported_type(self, processor, mock_dirs):
        input_dir = mock_dirs[0]
        test_file = input_dir / "test.pdf"
        test_file.touch()
        
        with patch.object(processor.tool, 'is_supported_file_type', return_value=False):
            success, error = processor.process_file(test_file)
            assert success is False
            assert "Unsupported file type" in error
            
            # Check error file
            error_file = processor.error_dir / "test_error.json"
            assert error_file.exists()

    def test_process_file_with_validation(self, processor, mock_dirs):
        processor.validation = True
        input_dir = mock_dirs[0]
        test_file = input_dir / "test.pdf"
        test_file.touch()
        
        with patch('src.invoice_batch_proccessor.validate_and_process') as mock_validate:
            mock_validate.return_value = (False, None, ["validation error"])
            
            success, error = processor.process_file(test_file)
            assert success is False
            assert "Validation errors" in error
            
            # Check error file
            error_file = processor.error_dir / "test_validation_error.json"
            assert error_file.exists()

    @pytest.mark.asyncio
    async def test_process_batch(self, processor, mock_dirs):
        input_dir = mock_dirs[0]
        
        # Create test files
        test_files = ["test1.pdf", "test2.pdf", "test3.pdf"]
        for file_name in test_files:
            (input_dir / file_name).touch()
            
        # Create files that should be ignored
        (input_dir / "test.prompt").touch()
        (input_dir / "test_ontology.py").touch()
        
        results = await processor.process_batch()
        
        assert len(results) == len(test_files)
        assert all(r["success"] for r in results)
        
        # Check output files
        for file_name in test_files:
            output_file = processor.output_dir / f"{file_name.split('.')[0]}_processed.json"
            assert output_file.exists()
