import inspect
import logging
import json
import os
from pathlib import Path
from typing import Type, Optional, Dict, Any, Union, List, get_args, get_origin
from pydantic import BaseModel, ValidationError
from decimal import Decimal
from datetime import datetime
import importlib.util
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OntologyValidationError(Exception):
    """Custom exception for ontology validation errors"""
    def __init__(self, message: str, errors: List[str] = None):
        super().__init__(message)
        self.errors = errors or []

class OntologyManager:
    """Class for loading ontologies and generating prompts."""
    
    @staticmethod
    def _is_main_ontology_class(cls: Type) -> bool:
        """Check if a class is likely to be the main ontology class."""
        if not (inspect.isclass(cls) and issubclass(cls, BaseModel) and cls != BaseModel):
            return False
            
        name_indicators = ['ontology', 'schema', 'model', 'document']
        has_indicator = any(indicator in cls.__name__.lower() for indicator in name_indicators)
        
        field_count = len(cls.model_fields) if hasattr(cls, 'model_fields') else 0
        
        return field_count > 0 and (has_indicator or field_count >= 5)



    

    # @staticmethod
    # def _get_type_description(annotation: Any, field_description: str = None, example: str = None, is_optional: bool = False) -> str:
    #     """Generate type description for a field annotation including description and example"""
    #     origin = get_origin(annotation)
    #     args = get_args(annotation)
        
    #     # Handle Optional types
    #     if origin is Union and type(None) in args:
    #         inner_type = next(arg for arg in args if arg != type(None))
    #         return OntologyManager._get_type_description(inner_type, field_description, example, is_optional=True)
        
    #     # Handle List types
    #     if origin is list:
    #         inner_type = args[0]
    #         return f"[{OntologyManager._get_type_description(inner_type, field_description, example)}]"
        
    #     # Handle Dict types
    #     if origin is dict:
    #         key_type, value_type = args
    #         return f"Dict[{OntologyManager._get_type_description(key_type, field_description, example)}, {OntologyManager._get_type_description(value_type, field_description, example)}]"
        
    #     # Handle basic types with descriptions and examples
    #     def format_type(type_str: str) -> str:
    #         desc = f" /* {field_description} */" if field_description else ""
    #         ex = f" /* Example: {example} */" if example else ""
    #         result = f"{type_str}{desc}{ex}".strip()
    #         return result + "?" if is_optional else result

    #     if annotation == str:
    #         return format_type("string")
    #     elif annotation in (int, Decimal):
    #         return format_type("number")
    #     elif annotation == bool:
    #         return format_type("boolean")
    #     elif annotation == datetime:
    #         return format_type("YYYY-MM-DD")
        
    #     # Handle Pydantic models
    #     if isinstance(annotation, type) and issubclass(annotation, BaseModel):
    #         model_structure = OntologyManager._get_model_structure(annotation)
    #         return model_structure + "?" if is_optional else model_structure
        
    #     base_type = str(annotation)
    #     return base_type + "?" if is_optional else base_type

    # @staticmethod
    # def _get_model_structure(model: Type[BaseModel], indent: int = 0) -> str:
    #     """Generate structure description for a Pydantic model with descriptions and examples"""
    #     fields = []
    #     indent_str = "\t" * indent
    #     next_indent_str = "\t" * (indent + 1)
        
    #     for name, field in model.model_fields.items():
    #         # Determine if field is optional
    #         is_optional = not field.is_required
            
    #         field_type = OntologyManager._get_type_description(
    #             field.annotation,
    #             field.description,
    #             field.json_schema_extra.get('example') if field.json_schema_extra else None,
    #             is_optional=is_optional
    #         )
            
    #         if isinstance(field_type, str) and field_type.startswith("{"):
    #             # For nested objects, handle the optionality in the field type itself
    #             fields.append(f'{next_indent_str}"{name}": {field_type}')
    #         elif isinstance(field_type, str) and field_type.startswith("[{"):
    #             field_type = field_type.replace("\n", f"\n{next_indent_str}")
    #             fields.append(f'{next_indent_str}"{name}": {field_type}')
    #         else:
    #             fields.append(f'{next_indent_str}"{name}": {field_type}')
        
    #     if not fields:
    #         return "{}"
        
    #     # Using doubled curly braces to escape them in the f-string
    #     return "{{\n{},\n}}".format(",\n".join(fields))
   
    @staticmethod
    def _get_type_description(annotation: Any, field_description: str = None, example: str = None, is_optional: bool = False) -> str:
        """Generate type description for a field annotation including description and example"""
        origin = get_origin(annotation)
        args = get_args(annotation)
        
        # Handle Optional types
        if origin is Union and type(None) in args:
            inner_type = next(arg for arg in args if arg != type(None))
            return OntologyManager._get_type_description(inner_type, field_description, example, is_optional=True)
        
        # Handle List types with special handling for Optional Lists
        if origin is list:
            inner_type = args[0]
            list_content = OntologyManager._get_type_description(inner_type, None, example)  # Don't pass description yet
            base_type = f"[{list_content}]"
            optional_marker = "?" if is_optional else ""
            desc = f" /* {field_description} */" if field_description else ""
            return f"{base_type}{optional_marker}{desc}"
        
        # Handle Dict types
        if origin is dict:
            key_type, value_type = args
            dict_content = f"Dict[{OntologyManager._get_type_description(key_type, None, example)}, {OntologyManager._get_type_description(value_type, None, example)}]"
            optional_marker = "?" if is_optional else ""
            desc = f" /* {field_description} */" if field_description else ""
            return f"{dict_content}{optional_marker}{desc}"
        
        # Handle basic types with descriptions and examples
        def format_type(type_str: str) -> str:
            optional_marker = "?" if is_optional else ""
            desc = f" /* {field_description} */" if field_description else ""
            ex = f" /* Example: {example} */" if example else ""
            return f"{type_str}{optional_marker}{desc}{ex}".strip()

        if annotation == str:
            return format_type("string")
        elif annotation in (int, Decimal):
            return format_type("number")
        elif annotation == bool:
            return format_type("boolean")
        elif annotation == datetime:
            return format_type("YYYY-MM-DD")
        
        # Handle Pydantic models
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            model_structure = OntologyManager._get_model_structure(annotation)
            optional_marker = "?" if is_optional else ""
            desc = f" /* {field_description} */" if field_description else ""
            return f"{model_structure}{optional_marker}{desc}"
        
        base_type = str(annotation)
        optional_marker = "?" if is_optional else ""
        desc = f" /* {field_description} */" if field_description else ""
        return f"{base_type}{optional_marker}{desc}"
        
    
    @staticmethod
    def _get_model_structure(model: Type[BaseModel], indent: int = 0) -> str:
        """Generate structure description for a Pydantic model with descriptions and examples"""
        fields = []
        indent_str = "\t" * indent
        next_indent_str = "\t" * (indent + 1)
        
        for name, field in model.model_fields.items():
            # Skip fields explicitly marked to be excluded
            if getattr(field, 'exclude', False):
                continue
                
            # Get field definition and metadata
            field_description = field.description
            field_example = field.json_schema_extra.get('example') if field.json_schema_extra else None
            is_optional = not field.is_required
            
            field_type = OntologyManager._get_type_description(
                field.annotation,
                field_description,
                field_example,
                is_optional=is_optional
            )
            
            # Handle different types of field structures
            if isinstance(field_type, str):
                if field_type.startswith("{"):
                    fields.append(f'{next_indent_str}"{name}": {field_type}')
                elif field_type.startswith("["):
                    field_type = field_type.replace("\n", f"\n{next_indent_str}")
                    fields.append(f'{next_indent_str}"{name}": {field_type}')
                else:
                    fields.append(f'{next_indent_str}"{name}": {field_type}')
        
        if not fields:
            return "{}"
        
        return "{{\n{}\n{:s}}}".format(",\n".join(fields), indent_str)



    @classmethod
    def load_ontology(cls, file_path: str) -> Type[BaseModel]:
        """Load ontology class from a file path."""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Import the module
            module_name = file_path.stem
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if not spec or not spec.loader:
                raise ImportError(f"Could not load module from {file_path}")
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find all classes in the module
            candidates: Dict[Type, int] = {}
            for name, obj in inspect.getmembers(module):
                if cls._is_main_ontology_class(obj):
                    score = 0
                    score += len(obj.model_fields) * 2
                    if 'ontology' in obj.__name__.lower():
                        score += 10
                    elif any(x in obj.__name__.lower() for x in ['schema', 'model', 'document']):
                        score += 5
                    if obj.__doc__:
                        score += 3
                    validators = [attr for attr in dir(obj) if 'validator' in attr.lower()]
                    score += len(validators)
                    candidates[obj] = score
            
            if not candidates:
                raise ValueError(f"No ontology class found in {file_path}")
            
            main_class = max(candidates.items(), key=lambda x: x[1])[0]
            logger.info(f"Found main ontology class: {main_class.__name__}")
            return main_class
            
        except Exception as e:
            logger.error(f"Error loading ontology from {file_path}: {str(e)}")
            raise

    @classmethod
    def load_ontology_by_class(cls, ontology_class: Type[BaseModel]) -> Type[BaseModel]:
        """Load and validate ontology from a class reference."""
        if not inspect.isclass(ontology_class):
            raise ValueError(f"Input must be a class, got {type(ontology_class)}")
            
        if not issubclass(ontology_class, BaseModel):
            raise ValueError(f"Class {ontology_class.__name__} must inherit from BaseModel")
            
        if not cls._is_main_ontology_class(ontology_class):
            raise ValueError(f"Class {ontology_class.__name__} is not a valid ontology class")
            
        return ontology_class

    @classmethod
    def validate_json(cls, ontology_class: Type[BaseModel], json_data: Union[str, dict]) -> tuple[bool, Optional[BaseModel], List[str]]:
        """
        Validate JSON data against the ontology with detailed error reporting.
        
        Args:
            ontology_class: The ontology class to validate against
            json_data: JSON string or dictionary to validate
            
        Returns:
            Tuple of (is_valid, validated_model, error_messages)
        """
        try:
            # Convert string to dict if needed
            if isinstance(json_data, str):
                try:
                    data_dict = json.loads(json_data)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error: {str(e)}")
                    return False, None, [f"Invalid JSON format: {str(e)}"]
            else:
                data_dict = json_data

            # Validate against the ontology
            try:
                validated_instance = ontology_class.model_validate(data_dict)
                return True, validated_instance, []
            except ValidationError as e:
                errors = []
                for error in e.errors():
                    # Get field location
                    location = " -> ".join(str(loc) for loc in error["loc"])
                    
                    # Get actual value that caused the error
                    input_value = error.get("input", "N/A")
                    
                    # Get expected type/constraint
                    expected = error.get("type", "unknown constraint")
                    
                    # Get specific error message
                    msg = error.get("msg", "unknown error")
                    
                    # Build detailed error message
                    error_msg = (
                        f"Field: {location}\n"
                        f"  Error: {msg}\n"
                        f"  Input value: {input_value}\n"
                        f"  Expected: {expected}\n"
                    )
                    
                    # Additional context for specific types of errors
                    if "Missing required field" in msg:
                        field_info = ontology_class.model_fields.get(location.split("->")[-1])
                        if field_info:
                            error_msg += f"  Field type: {field_info.annotation}\n"
                    
                    logger.error(error_msg)
                    errors.append(error_msg)
                
                # Add ontology field information for context
                errors.append("\nField requirements from ontology:")
                for field_name, field in ontology_class.model_fields.items():
                    required = "Required" if field.is_required else "Optional"
                    field_type = str(field.annotation)
                    errors.append(f"  {field_name}: {required}, Type: {field_type}")
                
                return False, None, errors

        except Exception as e:
            error_msg = (
                f"Unexpected error during validation: {str(e)}\n"
                f"Traceback:\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            return False, None, [error_msg]

    @classmethod
    def validate_and_process(cls, 
                           ontology_class: Type[BaseModel], 
                           json_data: Union[str, dict],
                           return_validated: bool = False,
                           verbose: bool = False) -> Union[tuple[bool, Optional[dict], List[str]], 
                                                        tuple[bool, Optional[BaseModel], List[str]]]:
        """
        Validate and optionally process JSON data against the ontology.
        
        Args:
            ontology_class: The ontology class to validate against
            json_data: JSON string or dictionary to validate
            return_validated: If True, returns validated Pydantic model instead of dict
            verbose: If True, includes additional validation details
            
        Returns:
            Tuple of (is_valid, data, error_messages)
            where data is either a dict or BaseModel depending on return_validated
        """
        is_valid, validated_model, errors = cls.validate_json(ontology_class, json_data)
        
        if not is_valid:
            if verbose:
                # Add field definitions for context
                errors.append("\nField Definitions:")
                for field_name, field in ontology_class.model_fields.items():
                    field_info = f"  {field_name}:\n"
                    field_info += f"    Type: {field.annotation}\n"
                    field_info += f"    Required: {field.is_required}\n"
                    if field.default is not None:
                        field_info += f"    Default: {field.default}\n"
                    errors.append(field_info)
            return is_valid, None, errors
            
        if return_validated:
            return True, validated_model, []
            
        # Convert to dict with proper JSON serialization
        try:
            json_dict = validated_model.model_dump()
            return True, json_dict, []
        except Exception as e:
            error_msg = f"Error converting to dict: {str(e)}\n{traceback.format_exc()}"
            return False, None, [error_msg]

    @classmethod
    def generate_prompt(cls, ontology_class: Type[BaseModel], model_type: int = 1) -> str:
        """Generate a prompt template based on the ontology structure, loading from a file based on the model type.

        Args:
            ontology_class: The Pydantic model representing the ontology.
            model_type:  1 for Gemini, 2 for Claude, 3 for GPT. Defaults to Gemini (1).

        Returns:
            The formatted prompt string.
        """
        # template_paths = {
        #     1: "gemini",  # Or any subdirectory you prefer
        #     2: "claude",
        #     3: "gpt"
        # }


        template_dir = os.path.dirname(__file__)  # Get the directory of the current script
        template_path = os.path.join(template_dir, "templates/", "gemini_ontology_template.txt")


        try:
            with open(template_path, "r") as f:
                template = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Template file not found at {template_path}")

        structure = cls._get_model_structure(ontology_class)
        prompt = template.format(structure=structure)
        return prompt
    
    

# Convenience functions
def load_ontology(file_path: str) -> Type[BaseModel]:
    """Load ontology class from a file path."""
    return OntologyManager.load_ontology(file_path)

def load_ontology_by_class(ontology_class: Type[BaseModel]) -> Type[BaseModel]:
    """Load and validate ontology from a class reference."""
    return OntologyManager.load_ontology_by_class(ontology_class)

def generate_prompt(ontology_class: Type[BaseModel]) -> str:
    """Generate prompt from an ontology class."""
    return OntologyManager.generate_prompt(ontology_class)

def validate_json(ontology_class: Type[BaseModel], json_data: Union[str, dict]) -> tuple[bool, Optional[BaseModel], List[str]]:
    """Validate JSON against ontology."""
    return OntologyManager.validate_json(ontology_class, json_data)

def validate_and_process(ontology_class: Type[BaseModel], 
                        json_data: Union[str, dict],
                        return_validated: bool = False,
                        verbose: bool = False) -> Union[tuple[bool, Optional[dict], List[str]], 
                                                     tuple[bool, Optional[BaseModel], List[str]]]:
    """
    Validate and process JSON against ontology.
    
    Args:
        ontology_class: The ontology class to validate against
        json_data: JSON string or dictionary to validate
        return_validated: If True, returns validated Pydantic model instead of dict
        verbose: If True, includes additional validation details
    """
    return OntologyManager.validate_and_process(
        ontology_class=ontology_class, 
        json_data=json_data, 
        return_validated=return_validated, 
        verbose=verbose
    )

# Example usage
if __name__ == "__main__":
    try:
        # Method 1: Load from file
        ontology_class = load_ontology("invoice_ontology.py")
        prompt1 = generate_prompt(ontology_class)
        
        # Method 2: Load from class reference
        from invoice_ontology import InvoiceOntology
        ontology_class = load_ontology_by_class(InvoiceOntology)
        prompt2 = generate_prompt(ontology_class)
        
        print(f"Successfully loaded {ontology_class.__name__}")
        print("\nGenerated Prompt:")
        print(prompt2)
        
    except Exception as e:
        print(f"Error: {str(e)}")