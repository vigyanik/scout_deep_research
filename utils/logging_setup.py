# utils/logging_setup.py

import logging
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from typing import Dict, Any, Optional, IO

# Import settings carefully, handle potential circularity or load errors early
from config.settings import Settings


# Configure root logger early
logging.basicConfig(level="INFO", format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', stream=sys.stderr)
logger = logging.getLogger(__name__)

# --- Standard Logging Setup ---
def setup_standard_logging(level: str = "INFO", log_to_console: bool = True):
    """
    Configures the standard Python logging framework.

    Args:
        level: The desired logging level (e.g., "DEBUG", "INFO").
        log_to_console: Whether to output standard logs to the console (stderr).
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    # Use basicConfig to configure the root logger simply
    # Force=True allows re-configuration if called multiple times
    logging.basicConfig(level=log_level, format=log_format, stream=sys.stderr if log_to_console else None, force=True)
    # Add file handler maybe? - Currently only logs to stderr if log_to_console is True
    # file_handler = logging.FileHandler("scout_debug.log")
    # file_handler.setLevel(log_level)
    # file_handler.setFormatter(logging.Formatter(log_format))
    # logging.getLogger().addHandler(file_handler) # Add to root logger

    logging.info(f"Standard logging initialized at level {level}.")


# --- JSON Logger Class ---
class JsonLogger:
    """
    Handles writing structured JSON logs to a file (JSON Lines format).
    """
    def __init__(self, log_file_path: str):
        """
        Initializes the logger and opens the log file.

        Args:
            log_file_path: The full path to the JSON Lines (.jsonl) log file.
        """
        self.log_file_path = log_file_path
        self._log_file: Optional[IO[str]] = None
        self._open_log_file()
        logger.info(f"JsonLogger initialized. Logging to: {self.log_file_path}")

    def _open_log_file(self):
        """Opens the log file in append mode with UTF-8 encoding."""
        try:
            # Ensure the directory exists
            log_dir = os.path.dirname(self.log_file_path)
            if log_dir: # Handle case where path might be just a filename
                 os.makedirs(log_dir, exist_ok=True)
            self._log_file = open(self.log_file_path, 'a', encoding='utf-8')
        except IOError as e:
            logging.error(f"Failed to open JSON log file '{self.log_file_path}': {e}", exc_info=True)
            self._log_file = None

    def _write_log(self, data: Dict[str, Any]):
        """Writes a JSON record to the log file if it's open."""
        if self._log_file and not self._log_file.closed:
            try:
                # Add a timestamp to every log entry
                data['timestamp_utc'] = datetime.now(timezone.utc).isoformat()
                # Use default=str for non-serializable types like Pydantic models or datetime
                json_record = json.dumps(data, ensure_ascii=False, default=str)
                self._log_file.write(json_record + '\n')
                self._log_file.flush() # Ensure it's written immediately
            except TypeError as e:
                 # Log specifically serialization errors
                 logging.error(f"JSON serialization error: {e}. Data causing error (partial): { {k: repr(v)[:100] for k, v in data.items()} }", exc_info=False) # Log short repr
            except IOError as e:
                 logging.error(f"Failed to write to JSON log file '{self.log_file_path}': {e}", exc_info=True)
                 self.close() # Close the potentially broken file handle
        else:
            logging.error(f"JSON log file not open or closed. Log entry lost: {data.get('log_type', 'UNKNOWN')}")

    def start_session(self, config_args: Dict[str, Any]) -> str:
        """Logs the start of a new session."""
        # Session ID based on timestamp + microseconds for uniqueness
        session_id = f"session_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        log_entry = {
            "log_type": "SESSION_START",
            "session_id": session_id,
            "config_args": config_args
        }
        self._write_log(log_entry)
        logger.info(f"JSON Logger: Started session {session_id}")
        return session_id

    def start_node(self, node_type: str, input_data: Dict[str, Any], parent_node_id: Optional[str] = None) -> str:
        """Logs the start of a processing node/step."""
        node_id = f"{node_type.lower()}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        log_entry = {
            "log_type": "NODE_START",
            "node_id": node_id,
            "parent_node_id": parent_node_id,
            "node_type": node_type,
            "input_data": input_data,
             "status": "STARTED" # Add initial status
        }
        self._write_log(log_entry)
        logger.debug(f"JSON Logger: Started node {node_id} (Type: {node_type}, Parent: {parent_node_id})")
        return node_id

    def end_node(self, node_id: str, status: str, output_data: Dict[str, Any], error_message: Optional[str] = None):
        """Logs the end of a processing node/step."""
        log_entry = {
            "log_type": "NODE_END",
            "node_id": node_id,
            "status": status.upper(), # Ensure status is uppercase (e.g., COMPLETED, ERROR)
            "output_data": output_data
        }
        if error_message:
            log_entry["error_message"] = error_message
        self._write_log(log_entry)
        logger.debug(f"JSON Logger: Ended node {node_id} with status {status}")


    def log_agent_call_start(self, node_id: str, agent_name: str, prompt: Dict[str, Any]) -> str:
        """Logs the start of an agent call."""
        call_id = f"call_{agent_name}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        log_entry = {
            "log_type": "AGENT_CALL_START",
            "call_id": call_id,
            "node_id": node_id,
            "agent_name": agent_name,
            "prompt": prompt, # Log the structured prompt
            "status": "STARTED" # Add initial status
        }
        self._write_log(log_entry)
        logger.debug(f"JSON Logger: Agent call start {call_id} (Agent: {agent_name}, Node: {node_id})")
        return call_id

    def log_agent_call_end(
            self,
            call_id: str,
            status: str,
            raw_response_data: Any, # Raw response from API/Agent
            parsed_output_data: Optional[Dict[str, Any]], # Structured data extracted
            error_message: Optional[str] = None,
            display_response: Optional[str] = None # Formatted for UI display (e.g., Markdown)
        ):
        """Logs the end of an agent call including raw, parsed, and display outputs."""

        # Create a loggable preview (limit size)
        loggable_response_preview = None
        if raw_response_data is not None:
            try:
                # Try dumping if it's a Pydantic model
                if hasattr(raw_response_data, 'model_dump_json'):
                    preview_str = raw_response_data.model_dump_json(exclude_unset=True, indent=2)
                # Try dumping if it's a dict/list
                elif isinstance(raw_response_data, (dict, list)):
                     preview_str = json.dumps(raw_response_data, default=str, indent=2)
                else: # Fallback to string representation
                    preview_str = str(raw_response_data)

                # Truncate preview if too long
                max_preview_len = 5000
                if len(preview_str) > max_preview_len:
                    loggable_response_preview = preview_str[:max_preview_len] + "..."
                else:
                    loggable_response_preview = preview_str
            except Exception as e:
                 logger.warning(f"Could not create preview for raw_response_data: {e}")
                 loggable_response_preview = repr(raw_response_data)[:1000] + "..." # Fallback repr


        log_entry = {
            "log_type": "AGENT_CALL_END",
            "call_id": call_id,
            "status": status.upper(), # Ensure status is uppercase
            "raw_response_preview": loggable_response_preview, # Log truncated preview
            # Log potentially large/complex data structures - handle serialization errors
            "raw_response_data": raw_response_data,
            "parsed_output_data": parsed_output_data,
            "display_response": display_response # Log the display version
        }
        if error_message:
            log_entry["error_message"] = error_message

        self._write_log(log_entry)
        logger.debug(f"JSON Logger: Agent call end {call_id} with status {status}")


    def log_event(self, event_type: str, data: Dict[str, Any], node_id: Optional[str] = None):
        """Logs a general event (e.g., error, info)."""
        log_entry = {
            "log_type": "EVENT",
            "event_type": event_type.upper(), # Standardize event type case
            "node_id": node_id, # Optional node context
            "data": data
        }
        self._write_log(log_entry)
        logger.debug(f"JSON Logger: Logged event {event_type} (Node: {node_id})")

    def close(self):
        """Closes the log file handle."""
        if self._log_file and not self._log_file.closed:
            try:
                self._log_file.close()
                logger.info(f"JSON log file closed: {self.log_file_path}")
            except IOError as e:
                logging.error(f"Error closing JSON log file '{self.log_file_path}': {e}", exc_info=True)
        self._log_file = None # Ensure handle is None after closing

    # Allow use as context manager
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Log uncaught exception if occurred within the 'with' block
        if exc_type:
             # Log simplified error to JSON log
             self.log_event(
                 event_type="FATAL_UNHANDLED_ERROR",
                 data={
                     "error_type": str(exc_type.__name__),
                     "error_message": str(exc_val),
                     # Optionally include limited traceback if needed, be careful with size
                     # "traceback": traceback.format_exception_only(exc_type, exc_val)[-1]
                 }
             )
             # Log full traceback to standard logger
             logger.critical(f"Unhandled exception caught by JsonLogger context manager: {exc_val}", exc_info=(exc_type, exc_val, exc_tb))
        self.close() # Ensure file is closed

    def session_end(self, session_id: str, status: str, summary_data: Optional[Dict[str, Any]] = None):
        """Logs the end of a session with optional summary data."""
        log_entry = {
            "log_type": "SESSION_END",
            "session_id": session_id,
            "status": status.upper(),  # Ensure status is uppercase
            "summary_data": summary_data or {}
        }
        self._write_log(log_entry)
        logger.info(f"JSON Logger: Session {session_id} ended with status {status}")


# --- Global Setup Function ---
# Keep this function, it's called by main.py (and potentially ui_server.py)
def setup_logging(settings: Settings) -> JsonLogger:
    """
    Configures standard logging and initializes the JsonLogger based on settings.
    This should only be called ONCE at application startup.

    Args:
        settings: The application settings object.

    Returns:
        An initialized JsonLogger instance.
    """
    # Configure standard Python logger (might be redundant if already called, but safe with force=True)
    setup_standard_logging(level=settings.logging.level)

    # Prepare JSON log file path - This is now handled within run_scout_process or UI background task
    # log_file_name = get_timestamped_filename(settings.logging.log_file_prefix, "jsonl")
    # log_file_path = os.path.join(settings.log_directory_path, log_file_name)

    # --- IMPORTANT ---
    # This global setup should NOT create the run-specific JsonLogger anymore.
    # The JsonLogger should be instantiated within the specific run context (CLI or UI background task)
    # with the run-specific log file path.
    # Returning a dummy or placeholder might be needed if the old structure relied on it.
    # Or, refactor `main.py` and `ui_server.py` to handle JsonLogger creation themselves.

    logger.info("Global logging setup complete (standard logger configured). JSON logger instance needs run-specific path.")
    # Return a placeholder or handle differently based on refactoring.
    # For now, let's return None, and ensure callers handle this.
    # return JsonLogger(log_file_path) # OLD BEHAVIOR - DO NOT USE GLOBALLY
    return None # Placeholder - JsonLogger should be created per run