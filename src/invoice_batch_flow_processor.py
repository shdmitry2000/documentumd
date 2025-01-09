from re import Pattern
import re
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import asyncio
import logging
import json, os
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel
import aiofiles

from app.processors.document_processor import PatternBasedGroupingStrategy
from app.tools.ontology.ontologyLoader import load_ontology
from flows.invoices.inwoice_flows import *


project_root = os.path.abspath(os.path.join(os.path.dirname('.'), '..'))

logger = logging.getLogger(__name__)

class PatternBasedGroupingStrategy:
    """Base strategy for grouping related files based on patterns"""
    
    def __init__(self, base_pattern: str, file_type_patterns: Dict[str, Pattern]):
        self.base_pattern = re.compile(base_pattern)
        self.file_type_patterns = file_type_patterns
    
 
    
    def _get_base_name(self, file_name: str) -> str:
        match = self.base_pattern.match(file_name)
        return match.group(1) if match else file_name
    
    def _get_file_type(self, file_name: str) -> str:
        for file_type, pattern in self.file_type_patterns.items():
            if pattern.search(file_name):
                return file_type
        return 'main'
    
    def group_files(self, directory: Path) -> List[Dict[str, Path]]:
        files = list(directory.glob('*'))
        if not files:
            return []
            
        groups: Dict[str, Dict[str, Path]] = {}
        for file_path in files:
            base_name = self._get_base_name(file_path.stem)
            file_type = self._get_file_type(file_path.name)
            
            if base_name not in groups:
                groups[base_name] = {}
                
            groups[base_name][file_type] = file_path
            
        return [group for group in groups.values() if 'main' in group]

class TemplateOntologyGroupingStrategy(PatternBasedGroupingStrategy):
    """Strategy for grouping invoice files with their templates and ontologies"""
    
    def __init__(self):
        super().__init__(
            base_pattern=r"^(.+?)(?:\.prompt|_ontology\.py)?$",
            file_type_patterns={
                'ontology': re.compile(r'_ontology\.py$'),
                'prompt': re.compile(r'\.prompt$')
            }
        )


class ProcessingResult(BaseModel):
    """Result of processing a single document"""
    success: bool
    file_group: Dict[str, Path]
    result: Optional[Dict[str, Any]]
    error: Optional[str]
    validation_errors: Optional[List[str]]
    flow_name: Optional[str] = None 

class FileManager:
    """Handles file I/O operations"""
    
    def __init__(self, output_dir: Path, error_dir: Path):
        self.output_dir = output_dir
        self.error_dir = error_dir
        
    async def save_result(self, result: ProcessingResult) -> None:
        try:
            main_file = result.file_group.get('main')
            if not main_file:
                raise ValueError("No main file found in file group")
                
            base_name = main_file.stem
            
            suffix = result.flow_name
            # Always save the result if we have one, regardless of validation errors
            if result.result:
                output_path = self.output_dir / f"{base_name}_{suffix}.json"
                await self._save_json(result.result, output_path)
            
            # Handle validation errors separately from regular errors
            if result.validation_errors:
                validation_error_path = self.error_dir / f"{base_name}_{suffix}_validation_error.json"
                error_data = {
                    'error': {
                        'message': "Validation failed",
                        'validation_errors': result.validation_errors,
                        'timestamp': datetime.now().isoformat()
                    },
                    'result': result.result,  # Include the result even with validation errors
                    'file_group': {k: str(v) for k, v in result.file_group.items()}
                }
                await self._save_json(error_data, validation_error_path)
            
            # Handle regular errors
            elif result.error:
                error_path = self.error_dir / f"{base_name}_{suffix}_error.json"
                error_data = {
                    'file_group': {k: str(v) for k, v in result.file_group.items()},
                    'error': result.error,
                    'timestamp': datetime.now().isoformat()
                }
                if result.result:
                    error_data['partial_result'] = result.result
                await self._save_json(error_data, error_path)
                
        except Exception as e:
            logger.error(f"Error saving result: {str(e)}", exc_info=True)
            raise

    async def _save_json(self, data: Dict, file_path: Path) -> None:
        """Save data as JSON to the specified path"""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(data, indent=2, ensure_ascii=False, default=str))
            
    
        
class InvoiceProcessor:
    """Main processor that coordinates file grouping, flow execution and result management"""
    
    def __init__(
        self,
        output_dir: Union[str, Path],
        error_dir: Union[str, Path],
        grouping_strategy: PatternBasedGroupingStrategy,
        validation=True,
        default_ontology_class: Optional[type] = None,
        model_name: str = "gemini-1.5-flash-002",
        max_workers: int = 1,
    ):
        self.output_dir = Path(output_dir)
        self.error_dir = Path(error_dir)
        self.grouping_strategy = grouping_strategy
        self.default_ontology_class = default_ontology_class
        self.file_manager = FileManager(self.output_dir, self.error_dir)
        self.workflow = create_processing_flow()
        self.model_name = model_name
        self.max_workers = max_workers
        self.processor_name = "invoice_processor"
        self.validation = validation

    async def process_directory(self, input_dir: Union[str, Path]) -> List[ProcessingResult]:
        """Process all file groups in a directory"""
        input_path = Path(input_dir)
        results = []
        
        # Group files using the strategy
        file_groups = self.grouping_strategy.group_files(input_path)
        
        # Create a semaphore to limit concurrent processing
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def process_with_semaphore(file_group):
            async with semaphore:
                try:
                    result = await self.process_file_group(file_group)
                    # Save result immediately after processing
                    await self.file_manager.save_result(result)
                    logger.info(f"Processed and saved result for {file_group.get('main', 'unknown file')}")
                    return result
                except Exception as e:
                    logger.error(f"Error processing {file_group.get('main', 'unknown file')}: {str(e)}")
                    error_result = ProcessingResult(
                        success=False,
                        file_group=file_group,
                        result=None,
                        error=str(e),
                        validation_errors=None,
                        flow_name=flow_name
                    )
                    await self.file_manager.save_result(error_result)
                    return error_result

        # Create tasks for each file group
        tasks = [process_with_semaphore(file_group) for file_group in file_groups]
        
        # Process files concurrently with progress updates
        for completed_task in asyncio.as_completed(tasks):
            result = await completed_task
            results.append(result)
            # Print progress
            logger.info(f"Progress: {len(results)}/{len(file_groups)} files processed")
            
        return results

    async def process_file_group(self, file_group: Dict[str, Path]) -> ProcessingResult:
        """Process a single file group"""
        try:
            if not self._should_process_file(file_group['main']):
                logger.info(f"Skipping file: {file_group['main']}")
                return ProcessingResult(
                    success=False,
                    file_group=file_group,
                    result=None,
                    error="File skipped by filter",
                    validation_errors=None,
                    flow_name=flow_name
                )
            
            initial_state = ProcessingState(
                config=self._create_flow_config(file_group),
                result=None,
                error=None,
                validation_errors=None,
                metadata={
                    'processing_date': datetime.now().isoformat(),
                    'source_file': str(file_group['main'])
                },
                flow_name=flow_name
            )
            
            try:
                logger.info(f"Starting workflow for {file_group['main']}")
                final_state = self.workflow.invoke(initial_state)
                logger.info(f"Workflow completed for {file_group['main']}")
                
                # Ensure flow_name is present in the result
                flow_name_value = final_state.get("flow_name", flow_name)
                
                result = ProcessingResult(
                    success=not (final_state["error"] or final_state["validation_errors"]),
                    file_group=file_group,
                    result=final_state["result"],
                    error=final_state["error"],
                    validation_errors=final_state["validation_errors"],
                    flow_name=flow_name_value
                )
                
                return result
                
            except Exception as e:
                logger.error(f"Workflow execution error: {str(e)}", exc_info=True)
                raise
            
        except Exception as e:
            error_msg = f"Error processing {file_group.get('main', 'unknown file')}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return ProcessingResult(
                success=False,
                file_group=file_group,
                result=None,
                error=error_msg,
                validation_errors=None,
                flow_name=flow_name
            )

    def _should_process_file(self, file_path: Path) -> bool:
        """
        Determine if a file should be processed based on its characteristics.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            bool: True if the file is a supported document type (PDF, Word, or image)
        """
        # Define supported file extensions
        supported_extensions = {
            # Documents
            '.pdf', '.doc', '.docx', 
            # Images
            '.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'
        }
        
        print("_should_process_file:", file_path)
        return file_path.suffix.lower() in supported_extensions
    
    
    def _create_flow_config(self, file_group: Dict[str, Path]) -> FlowConfig:
        """Create configuration for processing flow"""
        return FlowConfig(
            files=DocumentFiles(
                main_file=str(file_group['main']),
                ontology_file=str(file_group.get('ontology')) if 'ontology' in file_group else None,
                prompt_file=str(file_group.get('prompt')) if 'prompt' in file_group else None
            ),
            return_json_only=True,
            default_ontology_class=self.default_ontology_class,
            validation=self.validation,
        )


def print_results_summary(results: List[ProcessingResult]) -> None:
    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful
    
    print(f"\nProcessing Summary:")
    print(f"Total processed: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
   
    
    
    if failed > 0:
        print("\nFailed documents:")
        for result in results:
            if not result.success:
                print(f"\nDocument: {result.file_group.get('main')}")
                if result.error:
                    print(f"Error: {result.error}")
                if result.validation_errors:
                    print("Validation errors:")
                    for error in result.validation_errors:
                        print(f"  - {error}")

async def main():
    
    success = load_dotenv("../.env")
    print(f"Load success: {success}")
    INPUT_DIR=os.getenv("INPUT_DIR")
    OUTPUT_DIR=os.getenv("OUTPUT_DIR")
    ERROR_DIR=os.getenv("ERROR_DIR")

    logger.info("Starting document processing from %s" % INPUT_DIR)
    
    input_dir = INPUT_DIR + "/invoices"
    logger.info("processing %s", input_dir)
    output_dir = OUTPUT_DIR + "/invoices"
    error_dir = output_dir
    
    processor = InvoiceProcessor(
        output_dir=output_dir,
        error_dir=error_dir,
        grouping_strategy=TemplateOntologyGroupingStrategy(),
        default_ontology_class=None,
        validation=True
    )
    
    results = await processor.process_directory(input_dir)
    print_results_summary(results)

if __name__ == "__main__":
    asyncio.run(main())