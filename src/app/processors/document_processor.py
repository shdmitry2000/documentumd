from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Pattern, Tuple, Optional, TypeVar, Generic, Union
import re,os 
import logging

import concurrent

from app.tools.ontology.ontologyLoader import validate_and_process

logger = logging.getLogger(__name__)
from datetime import datetime
import traceback

T = TypeVar('T')  



class Processor(Generic[T]):
    """Interface for all processors."""
    @abstractmethod
    def process(self, input: T) -> Tuple[bool, Optional[str]]:
        """
        Process the input and return success status and optional error message.
        
        Args:
            input: The input to process (type specified by generic parameter T)
            
        Returns:
            Tuple of (success: bool, error_message: Optional[str])
        """
        pass

class FileProcessor:
    """Base class for file processing operations"""
    
    def __init__(self, 
                 output_dir: Optional[str] = None,
                 error_dir: Optional[str] = None,
                 default_ontology_class: Optional[type] = None,
                 validation: bool = True):
        self.output_dir = Path(output_dir) if output_dir else None
        self.error_dir = Path(error_dir) if error_dir else None
        self.default_ontology_class = default_ontology_class
        self.validation = validation
        
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.error_dir:
            self.error_dir.mkdir(parents=True, exist_ok=True)
            
    def get_error_path(self, file_path: Path, processor_name: str, validation: bool = False) -> Path:
        """Generate path for error file"""
        error_name = (f"{file_path.stem}_{processor_name}_validation_error.json" 
                     if validation 
                     else f"{file_path.stem}_{processor_name}_error.json")
        return self.error_dir / error_name if self.error_dir else Path(error_name)
    
    def get_output_path(self, file_path: Path, processor_name: str) -> Path:
        """Generate path for output file"""
        return (self.output_dir / f"{file_path.stem}_{processor_name}_processed.json" 
                if self.output_dir 
                else Path(f"{file_path.stem}_{processor_name}_processed.json"))
    
    def save_result(self, result: Dict, file_path: Path) -> None:
        """Save result to a JSON file"""
        try:
            # Convert Path to string for os.makedirs
            file_path_str = str(file_path)
            directory = os.path.dirname(file_path_str)
            
            # Create directory if it doesn't exist
            if directory:
                os.makedirs(directory, exist_ok=True)
            
            with open(file_path_str, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"Saved result to {file_path}")
        except Exception as e:
            logger.error(f"Error saving result to {file_path}: {str(e)}")
            raise
            
    def validate_result(self, result: Dict, ontology_class: type) -> Optional[List[str]]:
        """Validate result against ontology class"""
        if not self.validation or not ontology_class:
            return None
            
        try:
            is_valid, _, errors = validate_and_process(
                ontology_class=ontology_class, 
                json_data=result, 
                verbose=True
            )
            if not is_valid:
                return errors
            return None
        except Exception as e:
            logger.error(f"Error during validation: {str(e)}")
            return [str(e)]

    def save_error(self, error_data: Dict, file_path: Path, error: Exception) -> None:
        """Save error information to file"""
        error_data.update({
            'error': {
                'message': str(error),
                'traceback': traceback.format_exc(),
                'timestamp': datetime.now().isoformat()
            }
        })
        self.save_result(error_data, file_path)


class FileGroupingStrategy(ABC):
    """Abstract base class for file grouping strategies"""
    
    @abstractmethod
    def group_files(self, files: List[Path]) -> Dict[str, Dict[str, Path]]:
        """Group files according to the strategy"""
        pass

class PatternBasedGroupingStrategy(FileGroupingStrategy):
    """Groups files based on regex patterns"""
    
    def __init__(self, 
                 base_pattern: str = r"^(.+?)(?:_[^_]+)*$",
                 file_type_patterns: Dict[str, Pattern] = None):
        """
        Initialize with patterns for grouping files.
        
        Args:
            base_pattern: Regex pattern to extract the base group name
            file_type_patterns: Dict mapping file type names to regex patterns
        """
        self.base_pattern = re.compile(base_pattern)
        self.file_type_patterns = file_type_patterns or {}
    
    def add_file_type(self, type_name: str, pattern: str):
        """Add a new file type pattern"""
        self.file_type_patterns[type_name] = re.compile(pattern)
    
    def group_files(self, files: List[Path]) -> Dict[str, Dict[str, Path]]:
        grouped_files: Dict[str, Dict[str, Path]] = {}
        
        for file in files:
            
            # if file.suffix == '.py'  a:
            #     continue
                
            base_match = self.base_pattern.match(file.stem)
            if not base_match:
                continue
                
            base_name = base_match.group(1)
            if base_name.endswith('_ontology'):
                base_name=base_name.replace('_ontology','')
            
            if base_name not in grouped_files:
                grouped_files[base_name] = {}
            
            # Try to match file against each type pattern
            file_type = None
            for type_name, pattern in self.file_type_patterns.items():
                if pattern.search(file.name):
                    file_type = type_name
                    break
            
            # If no specific type matched and not a .prompt file
            if not file_type and not file.suffix == '.prompt' and not file.suffix == '.py':
                file_type = 'main'
            elif file.suffix == '.prompt':
                file_type = 'prompt'
            elif file_type == 'ontology' :
                pass
            else:
                continue
            # elif file.suffix == '_ontology.py':
            #     file_type = 'ontology'
            # elif file.suffix == '.py':
                
                
            grouped_files[base_name][file_type] = file
        
        # Only keep groups that have a main file
        return {k: v for k, v in grouped_files.items() if 'main' in v}



         
class BatchProcessor:
    """Enhanced generic class for batch processing documents"""

    def __init__(self, 
                 input_dir: str, 
                 output_dir: str, 
                 error_dir: str,
                 processors: List[Processor], 
                 grouping_strategy: FileGroupingStrategy,
                 max_workers: int = 4):
        
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.error_dir = Path(error_dir)
        self.max_workers = max_workers
        self.processors = processors
        self.grouping_strategy = grouping_strategy
        
        # Create directories if they don't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.error_dir.mkdir(parents=True, exist_ok=True)
    
    def _group_files(self) -> Dict[str, Dict[str, Path]]:
        """Groups files using the configured grouping strategy."""
        files = [item for item in self.input_dir.iterdir() if item.is_file()]
        return self.grouping_strategy.group_files(files)
    
    def _process_file_wrapper(self, 
                            file_group: Dict[str, Path], 
                            processor: Processor) -> dict:
        """
        Wrapper for process_file to be used with thread pool.
        
        Args:
            file_group: Dictionary mapping file types to their paths
            processor: Processor instance to use
            
        Returns:
            Dictionary containing processing results
        """
        try:
            success, error = processor.process(file_group)
            return {
                'group_id': next(iter(file_group.values())).stem,  # Use first file's name as group ID
                'processor': processor.__class__.__name__,
                'success': success,
                'error': error,
                'files': [str(path) for path in file_group.values()]
            }
        except Exception as e:
            logging.error(f"Error in processor {processor.__class__.__name__}: {str(e)}")
            return {
                'group_id': next(iter(file_group.values())).stem,
                'processor': processor.__class__.__name__,
                'success': False,
                'error': str(e),
                'files': [str(path) for path in file_group.values()]
            }

    async def process_batch(self) -> List[dict]:
        """
        Process all files in input directory using a thread pool.
        
        Returns:
            List of dictionaries containing processing results
        """
        logging.info(f"Starting batch processing in {self.input_dir}")
        
        grouped_files = self._group_files()
        results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for file_group in grouped_files.values():
                if not file_group:  # Skip empty groups
                    continue
                    
                for processor in self.processors:
                    futures.append(
                        executor.submit(
                            self._process_file_wrapper, 
                            file_group, 
                            processor
                        )
                    )
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logging.error(f"Error processing future: {str(e)}")
                    results.append({
                        'success': False,
                        'error': str(e)
                    })
        
        # Log summary
        total = len(results)
        successful = sum(1 for r in results if r['success'])
        logging.info(f"Batch processing complete. {successful}/{total} files processed successfully")
        
        return results