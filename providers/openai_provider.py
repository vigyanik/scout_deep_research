#!/usr/bin/env python3
"""
OpenAI Provider Module

This module implements various OpenAI-based agents for different tasks:
- Basic text generation
- Web search with preview
- Web search with references
- Structured output generation
- Composite search
- Large text chunk merging
"""

# =============================================================================
# Standard Library Imports
# =============================================================================
import logging
import time
import threading
import json
from typing import Any, Type, Dict, List, Optional
import openai
from openai.types import CompletionUsage # For usage stats if needed

# =============================================================================
# Third-Party Imports
# =============================================================================
from pydantic import BaseModel, ValidationError


# =============================================================================
# Internal Imports
# =============================================================================
from .base_provider import AgentProvider
from config.settings import Settings, ModelProviderConfig
from models.schemas import SearchContent, SearchReference

# =============================================================================
# Logger Setup
# =============================================================================
logger = logging.getLogger(__name__)

# =============================================================================
# Utility Classes
# =============================================================================
class RateLimiter:
    """
    Thread-safe rate limiter implementation using time-based delays.
    
    Attributes:
        delay (float): Time delay between requests in seconds.
        lock (threading.Lock): Thread lock for synchronization.
        last_call (float): Timestamp of the last API call.
    """
    
    def __init__(self, rps: float = 1.0):
        """
        Initialize the rate limiter.
        
        Args:
            rps (float): Requests per second. Defaults to 1.0.
        """
        if rps <= 0:
            rps = 1.0
            logger.warning(f"Invalid RPS value provided. Defaulting to {rps} RPS.")
        self.delay = 1.0 / rps
        self.lock = threading.Lock()
        self.last_call = 0.0
        logger.debug(f"Initialized RateLimiter with {rps:.2f} RPS (delay: {self.delay:.2f}s)")

    def wait(self):
        """
        Blocks execution if necessary to maintain the configured rate limit.
        Uses thread-safe locking to ensure accurate rate limiting.
        """
        with self.lock:
            now = time.time()
            elapsed = now - self.last_call
            wait_time = self.delay - elapsed
            if wait_time > 0:
                logger.debug(f"Rate limiting: waiting {wait_time:.3f} seconds")
                time.sleep(wait_time)
            self.last_call = time.time()

# =============================================================================
# OpenAI Agent Implementations
# =============================================================================
class OpenAIAgent(AgentProvider):
    """
    Basic OpenAI agent for text generation using Chat Completions API.
    
    This agent handles simple text generation tasks without any special
    features like web search or structured output.
    """
    
    def __init__(self, settings: Settings, model_config: ModelProviderConfig):
        """
        Initialize the OpenAI agent.
        
        Args:
            settings: Global settings object containing API keys and configuration.
            model_config: Model-specific configuration.
        """
        super().__init__(settings, model_config)
        self.name = f"OpenAIAgent-{model_config.name}"
        logger.info(f"Initializing {self.name} with model '{model_config.name}'")
        
        api_key = settings.api_keys.openai.get_secret_value() if settings.api_keys.openai else None
        if not api_key:
            raise ValueError("OpenAI API key is missing in settings.")
            
        self.client = openai.OpenAI(api_key=api_key)
        self.rate_limiter = RateLimiter(self.agent_settings.rate_limit_rps)

    def run(self, input_text: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate a text response using the OpenAI Chat Completions API.
        
        Args:
            input_text: The input text to process.
            system_prompt: Optional system prompt to guide the model's behavior.
            
        Returns:
            The generated text response.
            
        Raises:
            Exception: If the API call fails or returns an error.
        """
        logger.info(f"{self.name} processing input (length: {len(input_text)})")
        self.rate_limiter.wait()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": input_text})
        
        try:
            logger.debug(f"Sending request to OpenAI Chat Completions API model '{self.model_config.name}'")
            response = self.client.chat.completions.create(
                model=self.model_config.name,
                messages=messages,
            )
            
            result_text = ""
            if response.choices and response.choices[0].message:
                result_text = response.choices[0].message.content or ""
                
            logger.debug(f"Received response from OpenAI API (length: {len(result_text)})")
            return result_text.strip()
            
        except openai.APIError as e:
            logger.error(f"{self.name} OpenAI API error: {e}", exc_info=True)
            raise Exception(f"{self.name} API error: {e}")
        except Exception as e:
            logger.error(f"{self.name} error: {e}", exc_info=True)
            raise Exception(f"{self.name} error during API call: {e}")

class OpenAISearchAgent(AgentProvider):
    """
    OpenAI agent using the built-in 'web_search_preview' tool via the Responses API.
    
    This agent performs web searches and returns the results as text,
    without extracting structured references.
    """
    
    def __init__(self, settings: Settings, model_config: ModelProviderConfig):
        """
        Initialize the OpenAI search agent.
        
        Args:
            settings: Global settings object containing API keys and configuration.
            model_config: Model-specific configuration.
        """
        super().__init__(settings, model_config)
        self.name = f"OpenAISearchAgent-{model_config.name}"
        logger.info(f"Initializing {self.name} with model '{model_config.name}' "
                   f"(using Responses API + web_search_preview)")
        
        api_key = settings.api_keys.openai.get_secret_value() if settings.api_keys.openai else None
        if not api_key:
            raise ValueError("OpenAI API key is missing in settings.")
            
        self.client = openai.OpenAI(api_key=api_key)
        self.rate_limiter = RateLimiter(self.agent_settings.rate_limit_rps)

    def run(self, input_text: str) -> str:
        """
        Generate a text response grounded with web search results.
        
        Args:
            input_text: The search query or prompt.
            
        Returns:
            The generated text response with web search results.
            
        Raises:
            Exception: If the API call fails or returns an error.
        """
        logger.info(f"{self.name} processing query with web search (length: {len(input_text)})")
        self.rate_limiter.wait()
        
        try:
            logger.debug(f"Sending request to OpenAI Responses API model '{self.model_config.name}' "
                        f"with web_search_preview tool")
            response = self.client.responses.create(
                model=self.model_config.name,
                input=input_text,
                tools=[{"type": "web_search_preview"}]
            )
            
            result_text = ""
            if hasattr(response, 'output_text') and response.output_text:
                result_text = response.output_text
            elif response.output:
                for item in response.output:
                    if item.type == 'message' and item.content:
                        for part in item.content:
                            if part.type == 'output_text' and part.text:
                                result_text += part.text + "\n"
            else:
                logger.warning(f"{self.name}: No output_text or output array found in response.")
                
            logger.debug(f"Received web search grounded response from OpenAI API "
                        f"(length: {len(result_text)})")
            return result_text.strip()
            
        except openai.APIError as e:
            logger.error(f"{self.name} OpenAI API error: {e}", exc_info=True)
            raise Exception(f"{self.name} API error: {e}")
        except Exception as e:
            logger.error(f"{self.name} error: {e}", exc_info=True)
            raise Exception(f"{self.name} error during API call: {e}")

class OpenAISearchRefAgent(AgentProvider):
    """
    OpenAI agent using 'web_search_preview' and extracting citation annotations.
    Returns SearchContent with grounded text and structured references based on url_citation annotations.
    """
    def __init__(self, settings: Settings, model_config: ModelProviderConfig):
        # --- (Initialization remains the same) ---
        super().__init__(settings, model_config)
        self.name = f"OpenAISearchRefAgent-{model_config.name}"
        logger.info(f"Initializing {self.name} with model '{model_config.name}' (using Responses API + web_search_preview + citations)")
        api_key = settings.api_keys.openai.get_secret_value() if settings.api_keys.openai else None
        if not api_key:
            raise ValueError("OpenAI API key is missing in settings.")
        self.client = openai.OpenAI(api_key=api_key)
        self.rate_limiter = RateLimiter(self.agent_settings.rate_limit_rps)

    def _parse_response(self, response) -> SearchContent:
        """
        Internal helper to parse the OpenAI response object based on user's example structure.

        Args:
            response: The response object from `client.responses.create`.

        Returns:
            A SearchContent object.

        Raises:
            ValueError: If the expected response structure is not found.
        """
        logger.debug("Parsing OpenAI Responses API response for text and url_citations.")
        text = ""
        references = []
        message = None

        if not response or not hasattr(response, 'output') or not response.output:
             logger.warning("Response object does not contain the expected 'output' array.")
             # Attempt fallback to output_text if available
             if hasattr(response, 'output_text') and response.output_text:
                  logger.warning("Using response.output_text as fallback, annotations will be missing.")
                  return SearchContent(text=response.output_text, references=[])
             else:
                  raise ValueError("Response object is missing 'output' array and 'output_text'.")


        # Find the first assistant message in the output array
        for item in response.output:
            if getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant":
                message = item
                break

        if message is None or not hasattr(message, 'content') or not message.content:
            raise ValueError("No assistant message with content found in the response output.")

        # Assume the primary text is in the first content part of type 'output_text'
        # Concatenate text from all output_text parts just in case
        full_text_parts = []
        for content_part in message.content:
             if getattr(content_part, "type", None) == "output_text":
                  part_text = getattr(content_part, "text", "")
                  full_text_parts.append(part_text)

                  # Extract references from annotations within this part
                  if hasattr(content_part, "annotations") and content_part.annotations:
                       logger.debug(f"Found {len(content_part.annotations)} annotations in text part.")
                       for ann in content_part.annotations:
                            if getattr(ann, "type", None) == "url_citation":
                                try:
                                    start_index = getattr(ann, 'start_index', None)
                                    end_index = getattr(ann, 'end_index', None)
                                    url = getattr(ann, 'url', None)

                                    # Check if essential attributes are present
                                    if start_index is None or end_index is None or url is None:
                                        logger.warning(f"Skipping url_citation annotation due to missing index or URL: {ann}")
                                        continue

                                    ref = SearchReference(
                                        start_index=start_index,
                                        end_index=end_index,
                                        # Use getattr for potentially missing title
                                        title=getattr(ann, 'title', '') or 'N/A',
                                        url=url
                                    )
                                    references.append(ref)
                                    logger.debug(f"Extracted reference: '{ref.title}' ({ref.url}) @ indices [{ref.start_index}-{ref.end_index}]")
                                except Exception as ann_err:
                                    logger.warning(f"Failed to parse url_citation annotation: {ann_err}. Annotation data: {ann}", exc_info=True)
                            # else: # Log other annotation types if needed
                            #    logger.debug(f"Ignoring non-url_citation annotation: {getattr(ann, 'type', 'Unknown')}")


        text = "\n".join(full_text_parts).strip() # Join parts with newline

        if not text and not references:
             logger.warning("Parsed response resulted in empty text and no references.")
        elif not text and references:
             logger.warning("Parsed response resulted in empty text but found references.")


        return SearchContent(text=text, references=references)


    def run(self, input_text: str) -> SearchContent:
        """
        Generates text grounded by web search and extracts structured references.

        Args:
            input_text: The search query or prompt.

        Returns:
            A SearchContent object containing the web-grounded text and extracted references.
        """
        logger.info(f"{self.name} processing query with web search for references (length: {len(input_text)})")
        self.rate_limiter.wait()

        tools_config = [{"type": "web_search_preview"}]

        try:
            logger.debug(f"Sending web search request for references to OpenAI Responses API model '{self.model_config.name}'")
            response = self.client.responses.create(
                model=self.model_config.name,
                input=input_text,
                tools=tools_config,
            )

            # Use the internal parsing method
            parsed_content = self._parse_response(response)

            logger.info(f"Received web search response. Text length: {len(parsed_content.text)}, References extracted: {len(parsed_content.references)}")
            return parsed_content

        except openai.APIError as e:
            logger.error(f"{self.name} OpenAI API error: {e}", exc_info=True)
            raise Exception(f"{self.name} API error: {e}")
        except ValueError as ve: # Catch parsing errors specifically
             logger.error(f"{self.name} parsing error: {ve}", exc_info=True)
             raise Exception(f"{self.name} failed to parse response: {ve}")
        except Exception as e:
            logger.error(f"{self.name} unexpected error: {e}", exc_info=True)
            raise Exception(f"{self.name} error during API call: {e}")

class OpenAIStructuredAgent(AgentProvider):
    """
    OpenAI agent for structured output using the client.beta.chat.completions.parse helper.
    Ensures the model output is parsed directly into the provided Pydantic schema.
    """
    def __init__(self, settings: Settings, model_config: ModelProviderConfig):
        super().__init__(settings, model_config)
        self.name = f"OpenAIStructuredAgent-{model_config.name}"
        logger.info(f"Initializing {self.name} with model '{model_config.name}' "
                   f"for structured output (using completions.parse)")
        api_key = settings.api_keys.openai.get_secret_value() if settings.api_keys.openai else None
        if not api_key:
            raise ValueError("OpenAI API key is missing in settings.")
             # Decide whether to raise an error immediately or let it fail at runtime
             # raise AttributeError("OpenAI().beta.chat.completions.parse not found.")

        self.client = openai.OpenAI(api_key=api_key)
        self.rate_limiter = RateLimiter(self.agent_settings.rate_limit_rps)

    def run(self, input_text: str, response_schema: Type[BaseModel]) -> Any:
        """
        Generates and parses structured output conforming to the response_schema
        using the OpenAI SDK's beta parse helper.

        Args:
            input_text: The user's input prompt.
            response_schema: The Pydantic model class defining the expected structure.

        Returns:
            A validated instance of the response schema.

        Raises:
            ValueError: If the model refuses the request or returns unexpected data.
            AttributeError: If the .parse() method is not available in the installed SDK.
            Exception: If the API call fails for other reasons.
        """
        logger.info(f"{self.name} processing input for structured output (length: {len(input_text)})")
        self.rate_limiter.wait()

        # Prepare messages for the chat completions endpoint
        messages = [
            {"role": "system", "content": f"You are an expert at structured data extraction. Extract information from the user query and structure it according to the '{response_schema.__name__}' schema. Respond ONLY with the structured data."},
            {"role": "user", "content": input_text}
        ]

        try:
            logger.debug(f"Sending structured request to OpenAI API model '{self.model_config.name}' "
                        f"using beta completions.parse helper")

            # Use the client.beta.chat.completions.parse method
            completion = self.client.beta.chat.completions.parse(
                model=self.model_config.name,
                messages=messages,
                response_format=response_schema, # Pass the Pydantic model class directly
                # Optional: add temperature, max_tokens etc. if needed by the parse helper
                # temperature=self.agent_settings.get("temperature", 0.1),
                # max_tokens=self.agent_settings.get("max_completion_tokens", 2048)
            )

            # Extract the message object
            if not completion.choices:
                 logger.error(f"{self.name}: Received response with no choices.")
                 raise ValueError("OpenAI response contained no choices.")

            message = completion.choices[0].message

            # Check for refusal
            if message.refusal:
                logger.error(f"{self.name}: Model refused the request: {message.refusal}")
                raise ValueError(f"Model refused request: {message.refusal}")

            # Access the parsed Pydantic object
            if not hasattr(message, 'parsed') or message.parsed is None:
                 # This shouldn't happen if .parse was successful and no refusal, but check defensively
                 logger.error(f"{self.name}: Response message lacks 'parsed' attribute despite successful call and no refusal. Raw message: {message}")
                 raise ValueError("Failed to get parsed structured data from the response.")

            # The .parsed attribute should already be the validated Pydantic model instance
            validated_data = message.parsed
            logger.debug("Successfully parsed structured response using SDK helper.")
            return validated_data

        except AttributeError as ae:
             # Catch if the .parse method doesn't exist
             if 'parse' in str(ae):
                  logger.error("AttributeError: 'client.beta.chat.completions' object has no attribute 'parse'. "
                               "Please check your OpenAI library version.", exc_info=True)
                  raise AttributeError("The .parse() method for structured output is not available. "
                                       "Ensure your OpenAI library is up-to-date and supports this beta feature.")
             else:
                  logger.error(f"{self.name} AttributeError: {ae}", exc_info=True)
                  raise # Re-raise other AttributeErrors
        except openai.APIError as e:
            # Handle API-specific errors
            logger.error(f"{self.name} OpenAI API error: {e.status_code} - {e.message}", exc_info=True)
            raise Exception(f"{self.name} API error: {e}")
        except Exception as e:
            # Handle other potential errors (network, validation errors *within* parse, etc.)
            logger.error(f"{self.name} unexpected error: {e}", exc_info=True)
            raise Exception(f"{self.name} error during structured API call: {e}")

class OpenAICompositeSearchAgent(AgentProvider):
    """
    Composite agent that performs web search followed by structured output generation.
    
    This agent combines the capabilities of OpenAISearchAgent and OpenAIStructuredAgent
    to first perform a web search and then generate a structured response based on the
    search results.
    """
    
    def __init__(self, settings: Settings, model_config: ModelProviderConfig):
        """
        Initialize the composite search agent.
        
        Args:
            settings: Global settings object containing API keys and configuration.
            model_config: Model-specific configuration.
        """
        super().__init__(settings, model_config)
        self.name = f"OpenAICompositeSearchAgent-{model_config.name}"
        logger.info(f"Initializing {self.name} with model '{model_config.name}'")
        
        search_model_config = settings.get_model_config('search')
        structured_model_config = settings.get_model_config('structured_extract')
        
        if (search_model_config.provider != 'openai' or 
                structured_model_config.provider != 'openai'):
            logger.warning(f"OpenAICompositeSearchAgent expects OpenAI sub-agents, "
                         f"but found providers: {search_model_config.provider}, "
                         f"{structured_model_config.provider}. Trying to proceed.")
                         
        self.search_agent = OpenAISearchAgent(settings, search_model_config)
        self.structured_agent = OpenAIStructuredAgent(settings, structured_model_config)

    def run(self, input_text: str, response_schema: Type[BaseModel]) -> Any:
        """
        Perform a web search and generate a structured response.
        
        Args:
            input_text: The search query or prompt.
            response_schema: The Pydantic model class defining the expected structure.
            
        Returns:
            A validated instance of the response schema.
            
        Raises:
            Exception: If either the search or structured generation fails.
        """
        logger.info(f"{self.name} processing input (length: {len(input_text)})")
        try:
            logger.debug("Performing web search phase")
            search_results_text = self.search_agent.run(input_text)
            logger.debug(f"Web search phase completed, results length: "
                        f"{len(search_results_text)}")
            
            structured_prompt = f"""
Based on the original input query and the following information retrieved from a web 
search, generate a structured response conforming to the required JSON schema.
Incorporate key information from the retrieved information into the structured format.

ORIGINAL INPUT QUERY:
{input_text}

RETRIEVED INFORMATION (from Web Search):
{search_results_text}

Generate a structured JSON response based on this combined information.
"""
            logger.debug("Performing structured response phase using web search results")
            return self.structured_agent.run(structured_prompt, 
                                               response_schema=response_schema)
        except Exception as e:
            logger.error(f"{self.name} error during composite execution: {str(e)}", 
                        exc_info=True)
            raise Exception(f"{self.name} failed: {str(e)}")

class OpenAIChunkMergeLargeTextAgent(AgentProvider):
    """
    Agent to merge large texts by generating the output in chunks using OpenAI.
    
    This agent handles large text generation by breaking it into manageable chunks
    and merging them together, with safeguards against infinite loops and duplicate
    content.
    """
    
    def __init__(self, settings: Settings, model_config: ModelProviderConfig):
        """
        Initialize the chunk merge agent.
        
        Args:
            settings: Global settings object containing API keys and configuration.
            model_config: Model-specific configuration.
        """
        super().__init__(settings, model_config)
        self.name = f"OpenAIChunkMergeAgent-{model_config.name}"
        self.max_output_tokens_per_chunk = self.agent_settings.merge_chunk_size
        self.max_iterations = self.agent_settings.merge_max_iterations
        
        logger.info(f"Initializing {self.name} with model '{model_config.name}'")
        logger.info(f"Max output tokens per chunk: {self.max_output_tokens_per_chunk}, "
                   f"Max iterations: {self.max_iterations}")
        
        api_key = settings.api_keys.openai.get_secret_value() if settings.api_keys.openai else None
        if not api_key:
            raise ValueError("OpenAI API key is missing in settings.")
            
        self.client = openai.OpenAI(api_key=api_key)
        self.rate_limiter = RateLimiter(self.agent_settings.rate_limit_rps)

    def _generate_output_in_chunks(self, full_prompt: str) -> str:
        """
        Generate output piece by piece using chat completions.
        
        Args:
            full_prompt: The complete prompt to process.
            
        Returns:
            The accumulated output text.
            
        Raises:
            Exception: If chunk generation fails.
        """
        accumulated_output = ""
        last_chunk = ""
        completion_marker = "###NO_MORE_CONTENT###"
        system_prompt = "You are an expert editor merging text content provided in the user prompt. Follow the instructions precisely."
        
        for i in range(1, self.max_iterations + 1):
            self.rate_limiter.wait()
            iteration_instructions = (
                "...\nNow, please produce the NEXT sequential portion...\n"
                "Once you have fully completed...output the exact string "
                f"\"{completion_marker}\"..."
            )
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_prompt},
                *([{"role": "assistant", "content": accumulated_output}] 
                  if accumulated_output else []),
                {"role": "user", "content": iteration_instructions}
            ]
            #import pprint
            #pprint.pprint(messages)
            
            logger.info(f"Requesting chunk {i}/{self.max_iterations}...")
            try:
                response = self.client.chat.completions.create(
                    model=self.model_config.name,
                    messages=messages,
                    max_tokens=self.max_output_tokens_per_chunk,
                    stop=[completion_marker]
                )
                #pprint.pprint(response)
                
                chunk = ""
                finish_reason = "unknown"
                if response.choices and response.choices[0].message:
                    chunk = (response.choices[0].message.content or "").strip()
                    finish_reason = response.choices[0].finish_reason or "unknown"
                    
                logger.debug(f"Raw Chunk {i} received (length: {len(chunk)}), "
                           f"finish_reason: {finish_reason}: '{chunk[:100]}...'")
                
                if finish_reason == "stop":
                    logger.info(f"Model stopped generating for chunk {i}.")
                if chunk:
                    accumulated_output += "\n" + chunk
                    break
                if not chunk and finish_reason != 'length':
                    logger.info(f"Model returned empty output for chunk {i}. "
                              f"Assuming completion.")
                    break
                if i > 1 and chunk == last_chunk and len(chunk) > 50:
                    logger.warning(f"Chunk {i} is identical to the previous chunk. "
                                f"Stopping.")
                    break
                    
                accumulated_output += "\n" + chunk
                last_chunk = chunk
                
                if len(accumulated_output) > self.agent_settings.get(
                        "max_total_merge_length", 500000):
                    logger.warning("Accumulated output exceeds length limit.")
                    break
                    
            except openai.APIError as e:
                logger.error(f"Error generating chunk {i}: {e}", exc_info=True)
                raise
            except Exception as e:
                logger.error(f"Unexpected error during chunk generation {i}: {e}", 
                           exc_info=True)
                raise
                
        if i >= self.max_iterations:
            logger.warning(f"Reached max iterations ({self.max_iterations}).")
        return accumulated_output.strip()

    def run(self, input_text: str) -> str:
        """
        Execute the chunked generation process.
        
        Args:
            input_text: The input text to process.
            
        Returns:
            The merged output text.
            
        Raises:
            Exception: If the chunked generation fails.
        """
        logger.info(f"{self.name} processing input for chunked merge "
                   f"(length: {len(input_text)})")
        try:
            merged_text = self._generate_output_in_chunks(input_text)
            logger.info(f"Successfully generated merged text via chunking "
                       f"(final length: {len(merged_text)})")
            return merged_text
        except Exception as e:
            logger.error(f"{self.name} error during chunked merge execution: {e}", 
                        exc_info=True)
            raise Exception(f"{self.name} error: {e}")

# =============================================================================
# Factory Function
# =============================================================================
def create_agent(
    agent_role: str,
    model: str,
    settings: Settings,
) -> AgentProvider:
    """
    Factory function to create a specific OpenAI agent instance based on role.
    
    Args:
        agent_role: The role of the agent to create.
        model: The model name to use.
        settings: Global settings object.
        
    Returns:
        An instance of the appropriate AgentProvider subclass.
        
    Raises:
        ValueError: If the agent role is unknown.
        Exception: If agent creation fails.
    """
    logger.info(f"OpenAI Factory: Creating agent with role '{agent_role}' "
               f"using model '{model}'")
    model_config = ModelProviderConfig(provider="openai", name=model)
    
    if agent_role.lower() == "structured":
        AgentClass = OpenAIStructuredAgent
    elif agent_role.lower() == "merge":
        AgentClass = OpenAIChunkMergeLargeTextAgent
    elif agent_role.lower() in ["default", "summary"]:
        AgentClass = OpenAIAgent
    elif agent_role.lower() == "search":
        AgentClass = OpenAISearchAgent
        logger.info("Mapping 'search' to OpenAISearchAgent (web search).")
    elif agent_role.lower() == "search_ref":
        AgentClass = OpenAISearchRefAgent
        logger.warning("Mapping 'search_ref' to OpenAISearchRefAgent "
                      "(web search, limited refs).")
    elif agent_role.lower() == "composite_search":
        AgentClass = OpenAICompositeSearchAgent
        logger.info("Mapping 'composite_search' to OpenAICompositeSearchAgent "
                   "(web search -> structured).")
    else:
        logger.error(f"Unknown agent role for OpenAI: {agent_role}")
        raise ValueError(f"Unknown agent role for OpenAI: {agent_role}")
        
    try:
        agent_instance = AgentClass(settings=settings, model_config=model_config)
        logger.info(f"OpenAI Factory: Successfully created {AgentClass.__name__} "
                   f"for role '{agent_role}'")
        return agent_instance
    except Exception as e:
        logger.error(f"Failed to instantiate {AgentClass.__name__} for role "
                    f"'{agent_role}': {e}", exc_info=True)
        raise Exception(f"Failed to create agent for role '{agent_role}': {e}")