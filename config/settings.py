# config/settings.py

import logging # <--- Added import
import os
import sys # <--- Added import for sys.exit
import json
from pydantic import Field, BaseModel, SecretStr, DirectoryPath, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, Dict, Literal, Any

# --- Logger Setup ---
logger = logging.getLogger(__name__) # <--- Added definition

# Helper function
def get_project_root() -> str:
    """Returns the absolute path to the project root directory."""
    # Assumes settings.py is in scout/config/
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Nested Configuration Models ---
class ApiKeysSettings(BaseModel):
    """Stores API keys loaded directly from config.json."""
    gemini: Optional[SecretStr] = None # Loaded from JSON key "gemini"
    openai: Optional[SecretStr] = None # Loaded from JSON key "openai"

class ModelProviderConfig(BaseModel):
    """Specifies the provider and model name for a specific task."""
    provider: Literal["gemini", "openai"] # Restrict to known providers
    name: str # Model name string (e.g., "gemini-1.5-flash", "gpt-4")

class ModelSettings(BaseModel):
    """Defines model configurations for different task types with fallbacks."""
    default: ModelProviderConfig
    merge: Optional[ModelProviderConfig] = None
    structured_search: Optional[ModelProviderConfig] = None
    summary: Optional[ModelProviderConfig] = None
    structured_extract: Optional[ModelProviderConfig] = None
    search: Optional[ModelProviderConfig] = None # For basic search agent
    search_ref: Optional[ModelProviderConfig] = None # Explicitly add if needed, though might fall back
    clarification_questions: Optional[ModelProviderConfig] = None # For clarification questions agent

    @model_validator(mode='after')
    def set_fallbacks(self) -> 'ModelSettings':
        """Ensure specific task models fall back to default if not provided."""
        self.merge = self.merge or self.default
        self.structured_search = self.structured_search or self.default
        self.summary = self.summary or self.default
        self.structured_extract = self.structured_extract or self.default
        self.search = self.search or self.default
        self.search_ref = self.search_ref or self.default # Add fallback for search_ref
        self.clarification_questions = self.clarification_questions or self.default # Add fallback for clarification_questions
        return self

class AgentSettings(BaseModel):
    """General settings applicable to agent behavior."""
    rate_limit_rps: float = Field(1.0, gt=0) # Requests per second (must be > 0)
    merge_chunk_size: int = Field(8000, gt=0) # For chunk merge agent
    merge_max_iterations: int = Field(20, gt=0) # For chunk merge agent

class LoggingSettings(BaseModel):
    """Configuration for logging."""
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_dir: str = "run_logs" # Relative path to project root
    log_file_prefix: str = "run" # Prefix for JSONL log files

class OutputSettings(BaseModel):
    """Configuration for output files."""
    base_dir: str = "output_reports" # Relative path to project root


# --- Main Settings Class ---
class Settings(BaseSettings):
    """Loads and validates application configuration."""
    # Configure Pydantic-settings: primarily load from JSON, ignore .env/prefix for keys
    model_config = SettingsConfigDict(
        env_nested_delimiter='__', # Still allow nested env vars for non-key overrides
        extra='ignore' # Ignore environment variables that don't match fields
    )

    # Define configuration fields using the nested models
    api_keys: ApiKeysSettings = Field(default_factory=ApiKeysSettings)
    models: ModelSettings # Models section is required
    agents: AgentSettings = Field(default_factory=AgentSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    output: OutputSettings = Field(default_factory=OutputSettings)

    # Paths resolved during initialization
    log_directory_path: Optional[DirectoryPath] = None
    output_directory_path: Optional[DirectoryPath] = None

    def __init__(self, config_file_path: Optional[str] = None, **values):
        """
        Initializes settings, loading primarily from JSON config file.

        Args:
            config_file_path: Optional path to the JSON config file. Defaults to 'config/config.json'.
            **values: Additional values to override loaded config (used internally by BaseSettings).
        """
        # Determine config file path
        if config_file_path is None:
             project_root = get_project_root()
             self._config_file_path = os.path.join(project_root, 'config', 'config.json')
        else:
             self._config_file_path = config_file_path

        # Load JSON config first to act as the base
        json_config = self._load_json_config()

        # Initialize BaseSettings using the loaded JSON data.
        # Pydantic-settings will handle potential overrides from environment variables
        # for fields *other than* api_keys (since validation_alias was removed).
        super().__init__(**json_config)

        # Perform post-initialization steps
        self._resolve_paths()
        self._validate_api_keys()
        logger.info("Settings initialized successfully.")


    def _load_json_config(self) -> Dict:
        """Loads configuration from the specified JSON file."""
        logger.debug(f"Attempting to load configuration from: {self._config_file_path}")
        if os.path.exists(self._config_file_path):
            try:
                with open(self._config_file_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                    # Ensure api_keys section exists for validation later
                    if 'api_keys' not in config_data:
                        config_data['api_keys'] = {}
                    logger.debug("Successfully loaded config.json")
                    return config_data or {}
            except json.JSONDecodeError as e:
                logger.error(f"Could not parse config file '{self._config_file_path}': {e}", exc_info=True)
                raise ValueError(f"Invalid JSON in config file '{self._config_file_path}': {e}")
            except Exception as e:
                logger.error(f"Could not load config file '{self._config_file_path}': {e}", exc_info=True)
                raise IOError(f"Could not read config file '{self._config_file_path}': {e}")
        else:
             logger.error(f"Configuration file not found: {self._config_file_path}")
             raise FileNotFoundError(f"Configuration file not found: {self._config_file_path}")


    def _resolve_paths(self):
        """Resolves relative paths for log and output directories and creates them."""
        project_root = get_project_root()
        self.log_directory_path = os.path.join(project_root, self.logging.log_dir)
        self.output_directory_path = os.path.join(project_root, self.output.base_dir)
        try:
            os.makedirs(self.log_directory_path, exist_ok=True)
            os.makedirs(self.output_directory_path, exist_ok=True)
            logger.debug(f"Log directory resolved to: {self.log_directory_path}")
            logger.debug(f"Output directory resolved to: {self.output_directory_path}")
        except OSError as e:
            logger.error(f"Failed to create directories: {e}", exc_info=True)
            raise # Re-raise directory creation errors


    def _validate_api_keys(self):
        """Checks if necessary API keys (loaded from JSON) are present for providers used in models."""
        logger.debug("Validating presence of required API keys...")
        used_providers = set()
        if not hasattr(self, 'models') or not self.models:
             raise ValueError("Configuration error: 'models' section is missing or empty in config file.")

        # Iterate through all defined model configs using model_dump for safe access
        for task_name, model_config_dict in self.models.model_dump().items():
             # Check if it's a valid model config dict (handles potential nesting/metadata)
             if isinstance(model_config_dict, dict) and 'provider' in model_config_dict:
                 used_providers.add(model_config_dict['provider'])

        if not used_providers:
             logger.warning("No providers found in the 'models' configuration.")
             # Depending on requirements, this might be an error or just a warning
             # raise ValueError("No providers specified in the 'models' configuration.")
             return # Allow proceeding if no models explicitly use providers? Or error?

        logger.debug(f"Providers found in configuration: {used_providers}")

        # Check for keys based on used providers
        if "gemini" in used_providers:
            gemini_key = self.api_keys.gemini.get_secret_value() if self.api_keys and self.api_keys.gemini else None
            if not gemini_key:
                raise ValueError("Configuration error: Gemini provider is used by models, but 'gemini' API key is missing or null in config.json -> api_keys.")
            logger.debug("Gemini API key found.")

        if "openai" in used_providers:
            openai_key = self.api_keys.openai.get_secret_value() if self.api_keys and self.api_keys.openai else None
            if not openai_key:
                raise ValueError("Configuration error: OpenAI provider is used by models, but 'openai' API key is missing or null in config.json -> api_keys.")
            logger.debug("OpenAI API key found.")
        # Add checks for other providers if supported

        logger.debug("API key validation passed.")


    def get_model_config(self, task: Literal["default", "merge", "structured_search", "summary", "structured_extract", "search", "search_ref"]) -> ModelProviderConfig:
         """
         Safely retrieves the model configuration for a given task name.
         Relies on the fallback logic defined in the ModelSettings validator.
         """
         if not hasattr(self.models, task):
              # This should generally not happen if ModelSettings includes all keys or uses __root__ appropriately
              # But adding a check for safety.
              logger.error(f"Configuration access error: Task '{task}' not found in settings.models.")
              raise AttributeError(f"Task '{task}' is not a valid model configuration key.")
         config = getattr(self.models, task)
         if config is None:
              # This might happen if default is None and no specific config is set - validation should prevent this.
              logger.error(f"Configuration error: Effective model config for task '{task}' resolved to None.")
              raise ValueError(f"No valid model configuration found for task '{task}' after fallbacks.")
         return config


# --- Global Settings Instance ---
# Instantiate settings once on import, handling potential errors during load.
try:
    settings = Settings()
except (ValueError, FileNotFoundError, IOError, AttributeError) as e:
    # Catch specific, expected errors during initial settings load/validation
    logger.error(f"CRITICAL ERROR initializing settings: {e}", exc_info=True)
    # Print user-friendly error and exit
    print(f"\nCRITICAL CONFIGURATION ERROR: {e}", file=sys.stderr)
    print("Please check your 'config/config.json' file and ensure it exists, is valid JSON, and contains required sections/keys.", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    # Catch any other unexpected error during settings instantiation
    logger.error(f"CRITICAL UNEXPECTED ERROR initializing settings: {e}", exc_info=True)
    print(f"\nCRITICAL UNEXPECTED ERROR initializing settings: {e}", file=sys.stderr)
    sys.exit(1)