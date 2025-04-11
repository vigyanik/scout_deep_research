# utils/helpers.py

import os
import logging
import time
import threading
import json # Added for log parsing
from datetime import datetime, timezone
from typing import List, Set, Optional, TypeVar, Dict, Any, Tuple # Added Tuple

# Assuming Reference schema is defined here or imported if needed for type hints
from models.schemas import Reference

logger = logging.getLogger(__name__)

# --- File/Timestamp Utilities ---

def get_timestamped_filename(base_name: str, extension: str, timestamp_str: Optional[str] = None) -> str:
    """
    Generates a filename with an optional timestamp.
    If timestamp_str is provided, it's used. Otherwise, the current time is used.
    If base_name is empty, only timestamp and extension are used.

    Args:
        base_name: The base part of the filename (can be empty).
        extension: The file extension (without the dot).
        timestamp_str: An optional pre-formatted timestamp string (YYYYMMDD_HHMMSS).

    Returns:
        A filename string, e.g., "basename_20230101_120000.txt" or "20230101_120000.log".
    """
    if not timestamp_str:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    if base_name:
        return f"{base_name}_{timestamp_str}.{extension}"
    else:
        # Useful for log files where only timestamp matters
        return f"{timestamp_str}.{extension}"

# --- Text Processing Utilities ---

def clean_content_of_markdown_enclosure(content: str) -> str:
    """
    Removes surrounding markdown code fences (like ```markdown ... ``` or ``` ... ```)
    from a string, if present.

    Args:
        content: The input string potentially wrapped in markdown fences.

    Returns:
        The cleaned string without the fences.
    """
    content = content.strip()
    if content.startswith('```'):
        # Find the end of the opening fence line
        first_line_end = content.find('\n')
        if first_line_end == -1: # Handle single-line case like ```text```
             first_line_end = len(content)

        start_idx = first_line_end + 1 # Start after the first line break

        # Find the last closing fence ```
        end_idx = content.rfind('```')

        if end_idx > start_idx: # Ensure closing fence is after opening content
             # Check if the line before the end fence is empty or just whitespace
             line_before_end = content.rfind('\n', 0, end_idx)
             if line_before_end != -1 and content[line_before_end:end_idx].strip() == "":
                  end_idx = line_before_end # Adjust end index to exclude the newline before closing fence
             return content[start_idx:end_idx].strip()
        else:
             # Malformed fence? Return content after opening fence line if no proper closing found
             logger.warning("Markdown fence detected but closing fence '```' seems missing or misplaced.")
             if content.startswith('```markdown') or content.startswith('```'):
                  return content[first_line_end:].strip()

    # No fences detected or handled, return original stripped content
    return content


# --- Data Structure Utilities ---

# Define a type variable for Reference-like objects that have a 'url' attribute
T = TypeVar('T', bound=Any) # Use Any as bound if Reference schema isn't directly used

def deduplicate_references(references: List[T]) -> List[T]:
    """
    Removes duplicate references from a list based on the 'url' attribute.
    Logs a warning if an item lacks a 'url' attribute or if the URL is empty.

    Args:
        references: A list of objects, expected to have a 'url' attribute (like models.schemas.Reference).

    Returns:
        A list containing only unique references based on their URL.
    """
    seen_urls: Set[str] = set()
    unique_references: List[T] = []
    for i, ref in enumerate(references):
        url = getattr(ref, 'url', None)
        if url and isinstance(url, str) and url.strip():
            url = url.strip() # Ensure consistent comparison
            if url not in seen_urls:
                unique_references.append(ref)
                seen_urls.add(url)
            # else: logger.debug(f"Duplicate reference skipped: {url}") # Optional: log duplicates
        else:
            logger.warning(f"Skipping reference at index {i} due to missing or empty URL: {ref}")
    return unique_references


# --- Concurrency Utilities ---

class RateLimiter:
    """
    A simple thread-safe rate limiter using a time delay between calls.
    """
    def __init__(self, rps: float = 1.0):
        """
        Initializes the rate limiter.

        Args:
            rps: Allowed requests per second. Must be positive. Defaults to 1.0.
        """
        if rps <= 0:
             rps = 1.0 # Default to 1 RPS if invalid value provided
             logger.warning(f"Invalid RPS value ({rps}) provided for RateLimiter. Defaulting to 1.0 RPS.")
        self.delay = 1.0 / rps
        self._lock = threading.Lock() # Use underscore for internal lock object
        self._last_call_time = 0.0
        logger.debug(f"Initialized RateLimiter with {rps:.2f} requests per second (delay: {self.delay:.3f}s)")

    def wait(self):
        """
        Blocks execution if necessary to maintain the configured rate limit.
        """
        with self._lock:
            now = time.monotonic() # Use monotonic clock for measuring intervals
            elapsed = now - self._last_call_time
            wait_time = self.delay - elapsed
            if wait_time > 0:
                logger.debug(f"Rate limiting: waiting {wait_time:.3f} seconds")
                time.sleep(wait_time)
            # Record call time *after* potential wait
            self._last_call_time = time.monotonic()


# --- Log Parsing Utilities (Adapted from log_viewer.py) ---

def parse_jsonl_log_data(jsonl_file_path):
    """
    Parse a JSONL log file and build structured data for visualization.
    
    Args:
        jsonl_file_path: Path to the JSONL log file
        
    Returns:
        dict: Dictionary containing nodes, tree structure, and session information
    """
    logger.debug(f"Parsing log data from {jsonl_file_path}")
    nodes = {}  # Will store node details by ID
    tree = []   # Will store tree nodes for jsTree format
    session_info = {}
    session_id = None  # Track session ID for parent references
    
    try:
        with open(jsonl_file_path, 'r') as f:
            for line in f:
                try:
                    log_entry = json.loads(line.strip())
                    event = log_entry.get("log_type")  # Changed from "event" to "log_type"
                    logger.debug(f"Processing log entry with type: {event}")
                    
                    # Process different event types
                    if event == "SESSION_START":
                        # Save session info and create a root node
                        session_info = log_entry.get("config_args", {})  # Changed from session_info to config_args
                        session_id = log_entry.get("session_id")
                        
                        if session_id:
                            # Create a root node for the session
                            session_node = {
                                "id": session_id,
                                "text": f"Research Session: {session_info.get('args', {}).get('research_topic_preview', 'No topic')}",  # Updated path to research topic
                                "type": "session",
                                "status": "RUNNING",
                                "start_time": log_entry.get("timestamp_utc"),  # Changed from timestamp to timestamp_utc
                                "session_info": session_info
                            }
                            nodes[session_id] = session_node
                            
                            # Add to tree structure
                            tree.append({
                                "id": session_id,
                                "text": f"Research Session",
                                "icon": "fas fa-project-diagram",
                                "parent": "#",  # Root node
                                "li_attr": {"class": "node-status-running"}
                            })
                    
                    elif event == "NODE_START":
                        node_id = log_entry.get("node_id")
                        node_type = log_entry.get("node_type", "unknown")
                        parent_id = log_entry.get("parent_node_id") or session_id  # Changed from parent_id to parent_node_id
                        
                        if node_id:
                            # Store node details
                            node_data = {
                                "id": node_id,
                                "text": f"{node_type} Node",  # Simplified text since node_name isn't in the schema
                                "type": node_type,
                                "status": "RUNNING",
                                "start_time": log_entry.get("timestamp_utc"),  # Changed from timestamp to timestamp_utc
                                "parent_id": parent_id,
                                "input_data": log_entry.get("input_data"),
                                "questions": log_entry.get("questions", [])  # Add questions for clarification nodes
                            }
                            nodes[node_id] = node_data
                            
                            # Map node type to icon class
                            icon_class = "fas fa-cog"  # Default icon
                            if "research" in node_type.lower():
                                icon_class = "fas fa-search"
                            elif "clarification" in node_type.lower():
                                icon_class = "fas fa-question-circle"
                            elif "merge" in node_type.lower():
                                icon_class = "fas fa-code-branch"
                                
                            # Add to tree structure
                            tree.append({
                                "id": node_id,
                                "text": node_data["text"],
                                "icon": icon_class,
                                "parent": parent_id or "#",  # Use parent_id or root if not available
                                "li_attr": {"class": "node-status-running"}
                            })
                    
                    elif event == "NODE_END":
                        node_id = log_entry.get("node_id")
                        status = log_entry.get("status", "COMPLETED")
                        
                        if node_id and node_id in nodes:
                            # Update node status
                            nodes[node_id]["status"] = status
                            nodes[node_id]["end_time"] = log_entry.get("timestamp_utc")  # Changed from timestamp to timestamp_utc
                            nodes[node_id]["output_data"] = log_entry.get("output_data")
                            
                            if status == "ERROR":
                                nodes[node_id]["error_message"] = log_entry.get("error_message")
                            
                            # Update tree node status class
                            status_class = "success" if status == "COMPLETED" else "error" if status == "ERROR" else "warning"
                            for tree_node in tree:
                                if tree_node["id"] == node_id:
                                    tree_node["li_attr"] = {"class": f"node-status-{status_class.lower()}"}
                                    break
                                    
                            # If this is the session node, update its status
                            if node_id == session_id:
                                # Session completed
                                for tree_node in tree:
                                    if tree_node["id"] == session_id:
                                        tree_node["li_attr"] = {"class": f"node-status-{status_class.lower()}"}
                                        break
                    
                    elif event == "AGENT_CALL_START":
                        call_id = log_entry.get("call_id")
                        parent_node_id = log_entry.get("node_id")
                        agent_name = log_entry.get("agent_name", "Unknown Agent")
                        
                        if call_id and parent_node_id:
                            # Initialize agent calls list if needed
                            if parent_node_id in nodes:
                                if "agent_calls" not in nodes[parent_node_id]:
                                    nodes[parent_node_id]["agent_calls"] = []
                                
                                # Add call details
                                call_data = {
                                    "call_id": call_id,
                                    "agent_name": agent_name,
                                    "status": "RUNNING",
                                    "start_time": log_entry.get("timestamp_utc"),  # Changed from timestamp to timestamp_utc
                                    "prompt": log_entry.get("prompt")
                                }
                                nodes[parent_node_id]["agent_calls"].append(call_data)
                                
                                # Optionally add agent call as a tree node
                                # This is a useful addition to show agent calls in the tree
                                tree.append({
                                    "id": call_id,
                                    "text": f"Agent Call: {agent_name}",
                                    "icon": "fas fa-robot",
                                    "parent": parent_node_id,
                                    "li_attr": {"class": "node-status-running"}
                                })
                                
                                # Also store in nodes for direct access
                                nodes[call_id] = {
                                    "id": call_id,
                                    "text": f"Agent Call: {agent_name}",
                                    "type": "agent_call",
                                    "status": "RUNNING",
                                    "start_time": log_entry.get("timestamp_utc"),  # Changed from timestamp to timestamp_utc
                                    "parent_id": parent_node_id,
                                    "agent_name": agent_name,
                                    "prompt": log_entry.get("prompt")
                                }
                    
                    elif event == "AGENT_CALL_END":
                        call_id = log_entry.get("call_id")
                        parent_node_id = log_entry.get("node_id")
                        status = log_entry.get("status", "SUCCESS")
                        
                        if call_id and parent_node_id and parent_node_id in nodes:
                            # Update agent call status
                            if "agent_calls" in nodes[parent_node_id]:
                                for call in nodes[parent_node_id]["agent_calls"]:
                                    if call["call_id"] == call_id:
                                        call["status"] = status
                                        call["end_time"] = log_entry.get("timestamp_utc")  # Changed from timestamp to timestamp_utc
                                        call["display_response"] = log_entry.get("display_response")
                                        call["parsed_output_data"] = log_entry.get("parsed_output_data")
                                        call["raw_response_data"] = log_entry.get("raw_response_data")
                                        
                                        if status == "FAILURE":
                                            call["error_message"] = log_entry.get("error_message")
                                        break
                            
                            # Update the tree node if we created one
                            status_class = "success" if status == "SUCCESS" else "error" if status == "FAILURE" else "warning"
                            for tree_node in tree:
                                if tree_node["id"] == call_id:
                                    tree_node["li_attr"] = {"class": f"node-status-{status_class.lower()}"}
                                    break
                                    
                            # Also update in direct nodes object
                            if call_id in nodes:
                                nodes[call_id]["status"] = status
                                nodes[call_id]["end_time"] = log_entry.get("timestamp_utc")  # Changed from timestamp to timestamp_utc
                                nodes[call_id]["display_response"] = log_entry.get("display_response")
                                nodes[call_id]["parsed_output_data"] = log_entry.get("parsed_output_data")
                                nodes[call_id]["raw_response_data"] = log_entry.get("raw_response_data")
                                
                                if status == "FAILURE":
                                    nodes[call_id]["error_message"] = log_entry.get("error_message")
                    
                    # Additional event types can be processed as needed
                    
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse line in log file: {line}")
                except Exception as e:
                    logger.error(f"Error processing log entry: {e}")
        
        # If no session ID was found, create a default root node
        if not session_id and tree:
            root_id = "root"
            session_id = root_id
            tree.insert(0, {
                "id": root_id,
                "text": "Research Session",
                "icon": "fas fa-project-diagram",
                "parent": "#",
                "li_attr": {"class": "node-status-info"}
            })
            nodes[root_id] = {
                "id": root_id,
                "text": "Research Session",
                "type": "session",
                "status": "UNKNOWN"
            }
            
            # Update parent references to point to the root
            for node in tree[1:]:
                if node["parent"] == "#":
                    node["parent"] = root_id
                    
        # Make sure all node IDs that appear in the tree are also in the nodes object
        # This ensures node details can be retrieved when clicking on tree nodes
        for tree_node in tree:
            node_id = tree_node["id"]
            if node_id not in nodes:
                # Create a basic node entry for any missing node
                nodes[node_id] = {
                    "id": node_id,
                    "text": tree_node["text"],
                    "type": "unknown",
                    "status": "UNKNOWN"
                }
        
        # Sort tree nodes to ensure parent nodes appear before children
        # This helps with proper tree rendering
        tree.sort(key=lambda x: 0 if x["parent"] == "#" else 1)
        
        logger.info(f"Successfully parsed log file. Found {len(nodes)} nodes and {len(tree)} tree items.")
        logger.debug(f"Node IDs: {list(nodes.keys())}")
        logger.debug(f"Tree structure: {[{'id': node['id'], 'parent': node['parent']} for node in tree]}")
        
        return {
            "nodes": nodes,
            "tree": tree,
            "session_info": session_info
        }
        
    except Exception as e:
        logger.error(f"Error parsing log file {jsonl_file_path}: {e}")
        return {
            "nodes": {},
            "tree": [],
            "session_info": {}
        }