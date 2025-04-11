# providers/gemini_provider.py

import logging
import time
import threading
import requests
from urllib.parse import urljoin
from typing import Any, Type, Dict # Added Dict

# Import BaseModel from Pydantic, needed for response_schema type hint
from pydantic import BaseModel

from google import genai
from google.genai import types as google_types # Avoid conflict with typing.types

# Internal project imports
from .base_provider import AgentProvider
from config.settings import Settings, ModelProviderConfig
# Import schemas used in this file directly
from models.schemas import SearchContent, SearchReference

# Set up logger for this module
logger = logging.getLogger(__name__)

# --- Utility Class (Consider moving to utils if used elsewhere) ---
class RateLimiter:
    def __init__(self, rps: float = 1.0):
        # Ensure rps is positive to avoid division by zero or negative delay
        if rps <= 0:
             rps = 1.0 # Default to 1 RPS if invalid value provided
             logger.warning(f"Invalid RPS value provided. Defaulting to {rps} RPS.")
        self.delay = 1.0 / rps
        self.lock = threading.Lock()
        self.last_call = 0.0
        logger.debug(f"Initialized RateLimiter with {rps:.2f} requests per second (delay: {self.delay:.2f}s)")

    def wait(self):
        with self.lock:
            now = time.time()
            elapsed = now - self.last_call
            wait_time = self.delay - elapsed
            if wait_time > 0:
                logger.debug(f"Rate limiting: waiting {wait_time:.3f} seconds")
                time.sleep(wait_time)
            self.last_call = time.time() # Record call time *after* potential wait

# --- Gemini Agent Implementations ---

class GeminiAgent(AgentProvider):
    """Basic Gemini agent for text generation."""
    def __init__(self, settings: Settings, model_config: ModelProviderConfig):
        super().__init__(settings, model_config)
        self.name = f"GeminiAgent-{model_config.name}"
        logger.info(f"Initializing {self.name} with model '{model_config.name}'")
        api_key = settings.api_keys.gemini.get_secret_value() if settings.api_keys.gemini else None
        if not api_key:
             raise ValueError("Gemini API key is missing in settings.")
        self.client = genai.Client(api_key = api_key) 
        self.rate_limiter = RateLimiter(self.agent_settings.rate_limit_rps)

    def run(self, input_text: str) -> str:
        logger.info(f"{self.name} processing input (length: {len(input_text)})")
        self.rate_limiter.wait()
        try:
            logger.debug(f"Sending request to Gemini API model '{self.model_config.name}'")
            response = self.client.models.generate_content( 
                model=self.model_config.name, 
                contents=[input_text]
            )
            result_text = response.text if hasattr(response, 'text') else ""
            logger.debug(f"Received response from Gemini API (length: {len(result_text)})")
            return result_text
        except Exception as e:
            logger.error(f"{self.name} error: {str(e)}", exc_info=True)
            raise Exception(f"{self.name} error during API call: {str(e)}")


class GeminiSearchAgent(AgentProvider):
    """Gemini agent using Google Search tool for grounding."""
    def __init__(self, settings: Settings, model_config: ModelProviderConfig):
        super().__init__(settings, model_config)
        self.name = f"GeminiSearchAgent-{model_config.name}"
        logger.info(f"Initializing {self.name} with model '{model_config.name}'")
        api_key = settings.api_keys.gemini.get_secret_value() if settings.api_keys.gemini else None
        if not api_key:
             raise ValueError("Gemini API key is missing in settings.")
        self.client = genai.Client(api_key = api_key)
        self.rate_limiter = RateLimiter(self.agent_settings.rate_limit_rps)

    def run(self, input_text: str) -> str:
        logger.info(f"{self.name} processing search query (length: {len(input_text)})")
        self.rate_limiter.wait()
        config = google_types.GenerateContentConfig(
            tools=[google_types.Tool(google_search=google_types.GoogleSearchRetrieval())]
        )
        try:
            logger.debug(f"Sending search request to Gemini API model '{self.model_config.name}'")
            response = self.client.models.generate_content( 
                model=self.model_config.name,
                contents=[input_text],
                config=config 
            )
            result_text = response.text if hasattr(response, 'text') else ""
            logger.debug(f"Received search response from Gemini API (length: {len(result_text)})")
            return result_text
        except Exception as e:
            logger.error(f"{self.name} error: {str(e)}", exc_info=True)
            raise Exception(f"{self.name} error during search API call: {str(e)}")


class GeminiSearchRefAgent(AgentProvider):
    """Gemini agent using Google Search and extracting references (original logic)."""
    def __init__(self, settings: Settings, model_config: ModelProviderConfig):
        super().__init__(settings, model_config)
        self.name = f"GeminiSearchRefAgent-{model_config.name}"
        logger.info(f"Initializing {self.name} with model '{model_config.name}'")
        api_key = settings.api_keys.gemini.get_secret_value() if settings.api_keys.gemini else None
        if not api_key:
             raise ValueError("Gemini API key is missing in settings.")
        self.client = genai.Client(api_key = api_key) # Use Client
        self.rate_limiter = RateLimiter(self.agent_settings.rate_limit_rps)
        self._url_cache: Dict[str, str] = {}

    # --- Helper methods exactly as in original provided code ---
    def _update_text_with_urls(self, search_content: SearchContent) -> SearchContent:
        # This method inserts the markdown links based on reference indices.
        logger.debug(f"Updating text with URLs for {len(search_content.references)} references")
        text = search_content.text
        # Sort references by end_index in descending order to avoid index shifts
        sorted_refs = sorted(search_content.references, key=lambda ref: ref.end_index, reverse=True)

        text_list = list(text) # Work with a list for efficient insertion
        for ref in sorted_refs:
             # Ensure title doesn't contain markdown breaking characters like ']'
             clean_title = ref.title.replace(']', '').replace('[', '')
             link_markdown = f"[{clean_title}]({ref.url})"
             # Insert markdown link. Handle potential index errors.
             if 0 <= ref.end_index <= len(text_list):
                 # Insert characters of the link one by one at the index
                 for char in reversed(link_markdown):
                      text_list.insert(ref.end_index, char)
             else:
                 logger.warning(f"Reference end_index {ref.end_index} out of bounds for text length {len(text_list)}. Appending link.")
                 # Append with space to avoid merging with last word
                 text_list.extend(list(f" {link_markdown}"))

        search_content.text = "".join(text_list)
        return search_content

    def _resolve_redirect_url(self, url: str) -> str:
        """ Resolves single-level redirects using HEAD request. Includes caching & basic retry."""
        # (Logic exactly as provided in the user's original code block)
        if not url or not url.startswith('http'):
             logger.debug(f"Skipping redirect check for invalid or non-HTTP URL: {url}")
             return url
        if url in self._url_cache:
             logger.debug(f"Using cached URL resolution for {url}")
             return self._url_cache[url]

        logger.info(f"Resolving potential single redirect for URL {url}")
        apply_redirect_check = True # Assume check all URLs as per original code comment

        if apply_redirect_check:
            max_retries = 3
            retry_count = 0
            last_exception = None
            # Using a session can be slightly more efficient for multiple requests
            session = requests.Session()

            while retry_count < max_retries:
                 try:
                     self.rate_limiter.wait()
                     logger.debug(f"Checking redirect for URL: {url} (attempt {retry_count + 1}/{max_retries})")
                     response = session.head(url, allow_redirects=False, timeout=5)

                     if response.is_redirect:
                         redirect_location = response.headers.get('Location')
                         if redirect_location:
                             final_url = urljoin(url, redirect_location)
                             logger.debug(f"Followed one redirect from {url} to: {final_url}")
                             self._url_cache[url] = final_url
                             return final_url
                         else:
                             logger.warning(f"Redirect status {response.status_code} for {url} but no 'Location' header. Returning original URL.")
                             final_url = url
                     else:
                         logger.debug(f"URL {url} is not a redirect (status: {response.status_code}).")
                         final_url = url

                     self._url_cache[url] = final_url
                     return final_url

                 except requests.exceptions.Timeout:
                     last_exception = "Timeout"
                     retry_count += 1
                     logger.warning(f"Attempt {retry_count}/{max_retries} timed out checking redirect for URL {url}")
                 except requests.exceptions.RequestException as e:
                     last_exception = e
                     retry_count += 1
                     logger.warning(f"Attempt {retry_count}/{max_retries} failed to check redirect for URL {url}: {str(e)}")
                 except Exception as e:
                     logger.error(f"An unexpected error occurred resolving {url}: {str(e)}", exc_info=True)
                     self._url_cache[url] = url
                     return url

                 if retry_count < max_retries:
                     time.sleep(1 * (2**(retry_count-1))) # Basic exponential backoff

            logger.error(f"All {max_retries} attempts failed to check redirect for URL {url}. Last error: {last_exception}")
            self._url_cache[url] = url
            return url

        else:
             logger.debug(f"URL {url} does not meet criteria for redirect check.")
             self._url_cache[url] = url
             return url


    def _parse_response(self, response) -> SearchContent:
        """ Parses Gemini API response using original grounding logic. """
        # (Logic exactly as provided in the user's original code block)
        logger.debug("Parsing Gemini API response using original grounding logic")
        full_text = ""
        references = []
        api_response_text = "" # Capture raw text just in case

        try:
            # Check for text, prefer parts if available
            if hasattr(response, 'text'):
                 api_response_text = response.text

            if not response.candidates:
                 logger.error("Response has no candidates.")
                 return SearchContent(text=api_response_text, references=[])

            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                 full_text = "".join(part.text for part in candidate.content.parts if hasattr(part, 'text'))
            else:
                 full_text = api_response_text # Fallback to raw text

            logger.debug(f"Extracted text length: {len(full_text)}")

            grounding_metadata = getattr(candidate, 'grounding_metadata', None)
            if grounding_metadata:
                grounding_chunks = getattr(grounding_metadata, 'grounding_chunks', []) or []
                grounding_supports = getattr(grounding_metadata, 'grounding_supports', []) or []
                logger.debug(f"Found {len(grounding_chunks)} grounding chunks and {len(grounding_supports)} grounding supports")

                for support in grounding_supports:
                    segment = getattr(support, 'segment', None)
                    if not segment or getattr(segment, 'start_index', None) is None or getattr(segment, 'end_index', None) is None:
                        logger.warning("Skipping support with missing segment indices")
                        continue

                    grounding_chunk_indices = getattr(support, 'grounding_chunk_indices', [])
                    for idx in grounding_chunk_indices:
                        if idx < len(grounding_chunks):
                            chunk = grounding_chunks[idx]
                            web_info = getattr(chunk, "web", None)
                            if web_info:
                                title = getattr(web_info, "title", "")
                                uri = getattr(web_info, "uri", "")
                                if uri:
                                    final_url = self._resolve_redirect_url(uri)
                                    reference = SearchReference(
                                        start_index=segment.start_index,
                                        end_index=segment.end_index,
                                        title=title,
                                        url=final_url
                                    )
                                    references.append(reference)
                                    logger.debug(f"Added reference: {title} ({final_url}) @ [{segment.start_index}-{segment.end_index}]")
                                else:
                                     logger.warning("Skipping grounding chunk web info with missing URI.")
                        else:
                            logger.warning(f"Grounding chunk index {idx} out of bounds (total chunks: {len(grounding_chunks)}).")
            else:
                 logger.debug("No grounding metadata found in candidate.")


            # Pass raw text and extracted references to SearchContent
            search_content = SearchContent(text=full_text, references=references)
            # Update text *after* parsing, using the helper method
            search_content_with_links = self._update_text_with_urls(search_content) # Use the helper
            logger.info(f"Successfully parsed response with {len(search_content_with_links.references)} references")
            return search_content_with_links

        except Exception as e:
             logger.error(f"Failed to parse Gemini response: {e}", exc_info=True)
             # Return raw text with empty references on parsing failure
             return SearchContent(text=api_response_text, references=[])


    def run(self, input_text: str) -> SearchContent:
        logger.info(f"{self.name} processing search query for references (length: {len(input_text)})")
        self.rate_limiter.wait()
        config = google_types.GenerateContentConfig(
            tools=[google_types.Tool(google_search=google_types.GoogleSearchRetrieval())]
        )
        try:
            logger.debug(f"Sending search request for references to Gemini API model '{self.model_config.name}' using original config method")
            response = self.client.models.generate_content( 
                model=self.model_config.name,
                contents=[input_text],
                config=config # Pass GenerationConfig object
                # Do NOT include tool_config or request_options here to match original request structure
            )
            logger.debug(f"Received response from Gemini API for reference search.")
            return self._parse_response(response) # Parse using original logic
        except Exception as e:
            logger.error(f"{self.name} error: {str(e)}", exc_info=True)
            raise Exception(f"{self.name} error during reference search API call: {str(e)}")


class GeminiStructuredAgent(AgentProvider):
    """Gemini agent for structured output (original logic)."""
    def __init__(self, settings: Settings, model_config: ModelProviderConfig):
        super().__init__(settings, model_config)
        self.name = f"GeminiStructuredAgent-{model_config.name}"
        # Original didn't use max_retries in init, removing it here too
        logger.info(f"Initializing {self.name} with model '{model_config.name}' for structured output")
        api_key = settings.api_keys.gemini.get_secret_value() if settings.api_keys.gemini else None
        if not api_key:
             raise ValueError("Gemini API key is missing in settings.")
        self.client = genai.Client(api_key = api_key) # Use Client
        self.rate_limiter = RateLimiter(self.agent_settings.rate_limit_rps)

    def run(self, input_text: str, response_schema: Type[BaseModel]) -> Any:
        """ Generates structured JSON using original config method. """
        logger.info(f"{self.name} processing input for structured output (length: {len(input_text)})")
        self.rate_limiter.wait()
        # Define generation_config as per original logic
        try:
            logger.debug(f"Sending structured request to Gemini API model '{self.model_config.name}' using original config method")
            response = self.client.models.generate_content( 
                model=self.model_config.name,
                contents=[input_text],
                config={
                        "response_mime_type": "application/json",
                        "response_schema": response_schema,
                }
            )

            # Parse using response.parsed as per original logic
            if not hasattr(response, 'parsed') or response.parsed is None:
                 # Handle cases where 'parsed' attribute might be missing or None
                 error_text = getattr(response, 'text', '(no text)')
                 logger.error(f"{self.name}: Response object lacks valid 'parsed' attribute. Response text: {error_text[:500]}")
                 # Attempt to parse text manually as a fallback, though original didn't show this
                 if error_text:
                      try:
                           import json
                           parsed_fallback = json.loads(error_text)
                           logger.warning(f"{self.name}: Manually parsed JSON from text as fallback.")
                           # Validate fallback data
                           validated_data = response_schema.model_validate(parsed_fallback)
                           return validated_data
                      except Exception as parse_err:
                           logger.error(f"{self.name}: Failed manual JSON parsing fallback: {parse_err}")
                 raise ValueError(f"{self.name}: Could not get parsed data from response.")

            validated_data = response.parsed # Directly use .parsed as per original code
            logger.debug("Received structured response from Gemini API (using response.parsed)")
            return validated_data

        except (ValueError, TypeError) as val_err: # Catch Pydantic validation errors if .parsed provides raw dict
             logger.error(f"{self.name} validation error: {str(val_err)}", exc_info=True)
             raise Exception(f"{self.name} failed to validate response against schema: {str(val_err)}")
        except Exception as e:
            # Catch potential API errors or issues during parsing/validation
            logger.error(f"{self.name} error: {str(e)}", exc_info=True)
            raise Exception(f"{self.name} error during structured API call: {str(e)}")


class GeminiCompositeSearchAgent(AgentProvider):
    """ Composite agent: search then structured output (using original structured logic). """
    def __init__(self, settings: Settings, model_config: ModelProviderConfig):
        super().__init__(settings, model_config)
        self.name = f"GeminiCompositeSearchAgent-{model_config.name}"
        logger.info(f"Initializing {self.name} with model '{model_config.name}'")

        # Get specific model configs for sub-tasks from settings
        # Allow different models for search vs structured if configured
        search_model_config = settings.get_model_config('search') # Assuming 'search' is a defined task in settings
        structured_model_config = settings.get_model_config('structured_extract') # Or relevant task

        # Ensure providers are Gemini for this composite agent
        if search_model_config.provider != 'gemini' or structured_model_config.provider != 'gemini':
             raise ValueError("GeminiCompositeSearchAgent currently only supports Gemini sub-agents.")

        # Instantiate sub-agents using their respective configs
        self.search_agent = GeminiSearchAgent(settings, search_model_config)
        # Use the GeminiStructuredAgent with reverted logic
        self.structured_agent = GeminiStructuredAgent(settings, structured_model_config)
        # Note: Original Composite agent directly instantiated sub-agents using raw params.
        # This refactored version uses the already defined agent classes.

    def run(self, input_text: str, response_schema: Type[BaseModel]) -> Any:
        logger.info(f"{self.name} processing input (length: {len(input_text)})")
        try:
            logger.debug("Performing search phase")
            search_results_text = self.search_agent.run(input_text)
            logger.debug(f"Search phase completed, results length: {len(search_results_text)}")

            structured_prompt = f"""
Based on original input text and following search results, generate a structured response according to the schema.
Make sure to incorporate key information from the search results into the structured format.

ORIGINAL INPUT TEXT:
{input_text}

SEARCH RESULTS:
{search_results_text}

Generate a structured response based on these results.
"""
            logger.debug("Performing structured response phase")
            # Call the structured agent (which uses the original logic now)
            return self.structured_agent.run(structured_prompt, response_schema=response_schema)

        except Exception as e:
            logger.error(f"{self.name} error during composite execution: {str(e)}", exc_info=True)
            raise Exception(f"{self.name} failed: {str(e)}")


class GeminiChunkMergeLargeTextAgent(AgentProvider):
    """ Agent to merge large texts by generating the output in chunks (original logic). """
    def __init__(self, settings: Settings, model_config: ModelProviderConfig):
        super().__init__(settings, model_config)
        self.name = f"GeminiChunkMergeAgent-{model_config.name}"
        self.max_output_tokens = self.agent_settings.merge_chunk_size
        self.max_iterations = self.agent_settings.merge_max_iterations
        logger.info(f"Initializing {self.name} with model '{model_config.name}'")
        logger.info(f"Max output tokens per chunk: {self.max_output_tokens}, Max iterations: {self.max_iterations}")
        api_key = settings.api_keys.gemini.get_secret_value() if settings.api_keys.gemini else None
        if not api_key:
             raise ValueError("Gemini API key is missing in settings.")
        self.client = genai.Client(api_key = api_key) # Use Client
        self.rate_limiter = RateLimiter(self.agent_settings.rate_limit_rps)

    def _generate_output_in_chunks(self, full_prompt: str) -> str:
        """ Internal method to generate output piece by piece (original logic). """
        accumulated_output = ""
        # Original had last_chunk = None, changing to "" for consistency in comparison
        last_chunk = ""
        completion_marker = "###NO_MORE_CONTENT###"

        for i in range(1, self.max_iterations + 1):
             self.rate_limiter.wait()

             # Iteration instructions as per original logic
             iteration_instructions = f"""
You are merging two pieces of text into one coherent article.
The full merge instructions are in the prompt below.
So far, you have produced this partial output (do NOT repeat it verbatim):

---PARTIAL OUTPUT SO FAR---
{accumulated_output}
---END PARTIAL OUTPUT---

Now, please produce the NEXT portion of the merged article.
Do not repeat any text already produced.
If you have fully completed the merged article and there is no more content to produce, just respond with "{completion_marker}" or an empty string.
"""
             prompt_for_this_chunk = full_prompt + "\n\n\n===========\n\n\n" + iteration_instructions

             logger.info(f"Requesting chunk {i}/{self.max_iterations} from model '{self.model_config.name}' (max_tokens: {self.max_output_tokens}).")

             try:
                 response = self.client.models.generate_content( 
                     model=self.model_config.name,
                     contents=[prompt_for_this_chunk],
                     config=google_types.GenerateContentConfig(
                            max_output_tokens=self.max_output_tokens
                        )
                 )

                 chunk = response.text.strip() if hasattr(response, 'text') else ""
                 logger.debug(f"Raw Chunk {i} received (length: {len(chunk)}): '{chunk[:100]}...'")

                 # Completion check exactly as per original logic
                 if completion_marker in chunk:
                     logger.info(f"Found completion marker '{completion_marker}' in chunk {i}.")
                     chunk = chunk.split(completion_marker)[0].strip()
                     if chunk:
                         accumulated_output += "\n" + chunk
                     break # Stop iterations

                 if not chunk:
                     logger.info(f"Model returned empty output for chunk {i}. Assuming completion.")
                     break

                 # Repetition check exactly as per original logic
                 if chunk == last_chunk:
                     logger.warning(f"Chunk {i} is identical to the previous chunk. Stopping iterations to prevent loop.")
                     break

                 accumulated_output += "\n" + chunk
                 last_chunk = chunk

                 # Add length check as a safety measure (optional, but good practice)
                 if len(accumulated_output) > 500000: # Example limit
                     logger.warning("Accumulated output exceeds length limit. Stopping merge.")
                     break

             except Exception as e:
                 logger.error(f"Error generating chunk {i}: {str(e)}", exc_info=True)
                 raise # raise on error

        if i == self.max_iterations and completion_marker not in chunk: # Adjusted condition slightly for clarity
            logger.warning(f"Reached max iterations ({self.max_iterations}) without finding completion marker or empty response.")

        return accumulated_output.strip()

    def run(self, input_text: str) -> str: # Changed return type hint to str as per original logic
        """ Executes the chunked generation process. """
        logger.info(f"{self.name} processing input for chunked merge (length: {len(input_text)})")
        try:
            # Generate the merged text using the internal method with original logic
            merged_text = self._generate_output_in_chunks(input_text)
            logger.info(f"Successfully generated merged text via chunking (final length: {len(merged_text)})")
            return merged_text
        except Exception as e:
            logger.error(f"{self.name} error during chunked merge execution: {str(e)}", exc_info=True)
            # Match original exception type/message
            raise Exception(f"{self.name} error: {str(e)}")


# providers/gemini_provider.py (Add this function at the end)

# ... (Keep all agent class definitions and imports as they were in the last version) ...

# --- Factory Function (Re-added based on original) ---
def create_agent(
    agent_role: str,
    model: str, # Model name string
    settings: Settings, # Pass settings for API keys, agent params etc.
    # Replacing original discrete params with settings access
    # additional_params: Dict[str, Any],
    # provider_keys: Dict[str, str],
    # rate_limit_rps: float
    ) -> AgentProvider:
    """
    Factory function to create a specific Gemini agent instance based on role.
    Uses the global Settings object for configuration.

    Args:
        agent_role: The conceptual role of the agent (e.g., 'search_ref', 'structured').
        model: The specific model name to use for this agent.
        settings: The application settings object.

    Returns:
        An initialized AgentProvider instance.

    Raises:
        ValueError: If the agent role is unknown or API key is missing.
        Exception: For other initialization errors.
    """
    logger.info(f"Gemini Factory: Creating agent with role '{agent_role}' using model '{model}'")

    # Construct the ModelProviderConfig dynamically for instantiation
    # Assuming the provider is always 'gemini' within this module's factory
    model_config = ModelProviderConfig(provider="gemini", name=model)

    # Map conceptual role to agent class
    if agent_role.lower() == "search":
        AgentClass = GeminiSearchAgent
    elif agent_role.lower() == "structured":
        AgentClass = GeminiStructuredAgent
    # The original composite agent was implicitly created in main.
    # If clarification needs 'composite_search' role, map it here.
    elif agent_role.lower() == "composite_search":
         AgentClass = GeminiCompositeSearchAgent
    elif agent_role.lower() == "search_ref":
        AgentClass = GeminiSearchRefAgent
    elif agent_role.lower() == "merge": # Renamed role from 'chunk_merge' for consistency maybe? Check main.
         # Assuming 'merge' role maps to the chunk merge agent
         AgentClass = GeminiChunkMergeLargeTextAgent
    elif agent_role.lower() == "default": # Role for basic agent
         AgentClass = GeminiAgent
    else:
        logger.error(f"Unknown agent role requested for Gemini provider: {agent_role}")
        raise ValueError(f"Unknown agent role for Gemini: {agent_role}")

    # Instantiate the determined class
    try:
         # Pass the dynamically created model_config
         agent_instance = AgentClass(settings=settings, model_config=model_config)
         logger.info(f"Gemini Factory: Successfully created {AgentClass.__name__} for role '{agent_role}'")
         return agent_instance
    except Exception as e:
         logger.error(f"Failed to instantiate agent class {AgentClass.__name__} for role '{agent_role}': {e}", exc_info=True)
         # Re-raise or handle more specifically
         raise Exception(f"Failed to create agent for role '{agent_role}': {e}")