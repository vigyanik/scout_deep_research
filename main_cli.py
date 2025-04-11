# scout/main_cli.py

import argparse
import json
import logging
import os
import sys
import time
import traceback
import importlib
from datetime import datetime
from typing import Any, Dict, Optional, List, Tuple
from pathlib import Path
from config.settings import settings
from models.schemas import SectionInput, Reference, QuestionList, SearchContent, FurtherResearch
from utils.logging_setup import setup_standard_logging, JsonLogger
from utils.helpers import get_timestamped_filename, deduplicate_references, clean_content_of_markdown_enclosure
from utils.formatter import write_output_files
from providers.base_provider import AgentProvider


# --- Add project root to path ---
project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
# --- End Path Addition ---




# Setup standard logger for the CLI
setup_standard_logging(level=settings.logging.level)
logger = logging.getLogger("scout_cli")


# --- Agent Creation Logic ---
def create_agent_cli(task_name: str) -> AgentProvider:
    """Creates an agent instance dynamically based on the task configuration."""
    try:
        model_config = settings.get_model_config(task_name)
        provider_name = model_config.provider
        model_name = model_config.name
    except AttributeError as e:
        logger.error(f"CLI: Failed to get model configuration for task '{task_name}': {e}", exc_info=True)
        raise ValueError(f"CLI: Could not load configuration for task: {task_name}")
    except Exception as e:
        logger.error(f"CLI: Unexpected error getting model configuration for task '{task_name}': {e}", exc_info=True)
        raise ValueError(f"CLI: Unexpected error loading configuration for task: {task_name}")

    role_map = {
        'default': 'default', 'search': 'search', 'search_ref': 'search_ref',
        'structured_extract': 'structured', 'merge': 'merge', 'summary': 'default',
        'structured_search': 'composite_search', 'clarification_questions': 'structured'
    }
    agent_role = role_map.get(task_name)
    if not agent_role:
        raise ValueError(f"CLI: No agent role mapping for task '{task_name}'.")

    module_path = f"providers.{provider_name}_provider"
    try:
        provider_module = importlib.import_module(module_path)
    except ImportError:
        raise ImportError(f"CLI: Could not import provider module '{module_path}'.")

    if not hasattr(provider_module, 'create_agent'):
        raise AttributeError(f"CLI: Provider module '{module_path}' lacks 'create_agent' factory.")

    factory_function = getattr(provider_module, 'create_agent')
    try:
        agent_instance = factory_function(agent_role=agent_role, model=model_name, settings=settings)
        logger.info(f"CLI: Created agent for task '{task_name}' (Role: {agent_role}, Provider: {provider_name}, Model: {model_name})")
        return agent_instance
    except Exception as e:
        raise Exception(f"CLI: Factory function failed for role '{agent_role}' in '{module_path}': {e}")


def run_clarification_phase(
    json_logger: JsonLogger,
    parent_node_id: str,
    run_id: str,
    clarification_agent: AgentProvider,
    summary_agent: AgentProvider,
    section_input: SectionInput,
) -> Optional[Dict[str, Any]]:
    """
    Clarification phase logic.
    Returns the refined context dictionary or None on failure/abort.
    """
    log_prefix = f"Run {run_id}"
    logger.info(f"{log_prefix}: Starting clarification phase.")
    clarification_node_id = json_logger.start_node(
        node_type="CLARIFICATION_NODE",
        input_data=section_input.model_dump(), # Log the input SectionInput model
        parent_node_id=parent_node_id
    )

    refined_context = section_input.model_dump() # Start with original input

    # 1. Get Questions
    questions = []
    prompt_dict_clarify = {
            "research_topic": section_input.research_topic, # Use the topic from input object
            "goal": "generate clarification questions",
            "role": "Research Assistant specializing in scope definition",
            "instructions": {
                # Updated instruction based only on topic
                "goal": "Analyze the provided research topic string and generate 2-4 questions (as strings) to help clarify the scope, priorities, specific inclusions/exclusions, or desired depth/focus for the research report based *only* on this topic.",
                "question_style": [
                    "Ask direct, close-ended questions that encourage the user to provide specific guidance. That is, describe the optionsand clarifications to choose from, and ask the user to select one or more or all.",
                    "Focus on areas that seem ambiguous or overly broad in the initial topic.",
                ],
                "output_format": f"Respond ONLY with a JSON object matching this schema: {QuestionList.model_json_schema()}. It contains a single key 'questions' holding a JSON list of strings.",
            }
    }
    questions_call_id = json_logger.log_agent_call_start(clarification_node_id, "clarification_agent", prompt=prompt_dict_clarify)
    q_status, q_raw, q_parsed, q_error, q_display = "FAILURE", None, None, None, None
    try:
        q_response = clarification_agent.run(json.dumps(prompt_dict_clarify), response_schema=QuestionList)
        q_raw = q_response
        if isinstance(q_response, QuestionList): questions = q_response.questions
        elif isinstance(q_response, dict) and 'questions' in q_response: questions = q_response['questions']
        else: raise ValueError(f"Unexpected clarification response: {type(q_response)}")

        if not questions:
            logger.warning(f"{log_prefix}: Clarification agent returned no questions.")
            q_status = "SUCCESS" # Successful call, just no questions
        else:
            q_status = "SUCCESS"
            q_parsed = {"questions": questions}
            q_display = f"Generated {len(questions)} questions."
            logger.info(f"{log_prefix}: Generated {len(questions)} questions.")
    except Exception as e:
        q_error = f"Clarification agent failed: {e}"
        logger.error(q_error, exc_info=True)
        json_logger.log_agent_call_end(questions_call_id, q_status, q_raw, q_parsed, q_error, q_display)
        json_logger.end_node(clarification_node_id, "ERROR", {}, q_error)
        return None # Failure
    finally:
        # Log end only if no exception was raised above
        if q_status == "SUCCESS":
             json_logger.log_agent_call_end(questions_call_id, q_status, q_raw, q_parsed, q_error, q_display)
    # 2. Get Answers (if questions were generated)
    answers = []
    if questions:
         try:
             answers = get_clarification_answers(questions)
             if answers is None: # Handle interruption
                  raise KeyboardInterrupt("User aborted clarification.")
         except KeyboardInterrupt:
             logger.warning(f"{log_prefix}: Clarification aborted by user.")
             json_logger.end_node(clarification_node_id, "ABORTED", {}, "User aborted.")
             return None # Signal abort
         except Exception as e:
             error_msg = f"Failed to get CLI answers: {e}"
             logger.error(f"{log_prefix}: {error_msg}", exc_info=True)
             json_logger.end_node(clarification_node_id, "ERROR", {}, error_msg)
             return None # Signal failure
    else:
        error_msg = f"Failed to get questions: {e}"
        logger.error(f"{log_prefix}: {error_msg}", exc_info=True)
        json_logger.end_node(clarification_node_id, "ERROR", {}, error_msg)
        return None # Signal failure  
     # 3. Generate Summary (if answers were provided)

    if answers:
        refined_context['clarifications'] = [{"question": q, "answer": a} for q, a in zip(questions, answers)]
        prompt_dict_summary = {
            "task": "summarize_research_plan",
            "role": "Research Report Writer",
            "instructions": {
                "goal": "Based on the initial research topic and any user-provided clarifications, generate a concise outline or summary of the planned research report.",
                "content": [
                    "Clearly state the main research topic.",
                    "Identify all key areas or top-level sections to be covered.",
                    "Incorporate any specific scope refinements from user clarifications.",
                ],
                "output_format": "Respond with a clear, easy-to-read plan outline or summary as plain text.",
            },
            "research_context": refined_context # Pass the dict which may include 'clarifications'
        }
        summary_call_id = json_logger.log_agent_call_start(clarification_node_id, "summary_agent", prompt={"refined_content": prompt_dict_summary, "num_answers": len(answers)})
        s_status, s_raw, s_parsed, s_error, s_display = "FAILURE", None, None, None, None
        try:
            summary_response = summary_agent.run(prompt_dict_summary)
            s_raw = summary_response
            if not isinstance(summary_response, str) or not summary_response.strip():
                raise ValueError("Summary agent returned invalid or empty response.")

            refined_context["plan_summary"] = summary_response.strip() # Store summary
            s_status = "SUCCESS"
            s_parsed = {"summary": summary_response.strip()}
            s_display = summary_response.strip()
            logger.info(f"{log_prefix}: Generated clarification summary.")

        except Exception as e:
            s_error = f"Summary agent failed: {e}"
            logger.error(f"{log_prefix}: {s_error}", exc_info=True)
            # Log end but allow proceeding without summary if desired, or return None to fail
            # For robustness, maybe proceed but log the error?
            json_logger.log_agent_call_end(summary_call_id, s_status, s_raw, s_parsed, s_error, s_display)
            # json_logger.end_node(clarification_node_id, "WARNING", refined_context, s_error) # End with warning?
            # return None # Fail the run if summary fails?
        finally:
            # Log end only if no exception or handled above
            if s_status == "SUCCESS":
                json_logger.log_agent_call_end(summary_call_id, s_status, s_raw, s_parsed, s_error, s_display)

    # Mark clarification node as completed if we got this far
    json_logger.end_node(clarification_node_id, "COMPLETED", refined_context, None)
    return refined_context


def get_clarification_answers(questions: List[str]) -> List[str]:
    """Get answers to clarification questions from the user."""
    logger.info("Starting to get clarification answers from user")
    logger.debug(f"Questions to ask: {questions}")

    answers = []
    try:
        for i, question in enumerate(questions, 1):
            logger.debug(f"Asking question {i}/{len(questions)}: {question}")
            print(f"\nQuestion {i}/{len(questions)}: {question}")
            answer = input("Your answer: ").strip()
            logger.debug(f"Received answer for question {i}: {answer}")

            if not answer:
                logger.warning(f"Empty answer received for question {i}")
                print("Please provide an answer.")
                answer = input("Your answer: ").strip()
                logger.debug(f"Received second attempt answer: {answer}")

            answers.append(answer)
            logger.debug(f"Added answer to list. Current answers: {answers}")

    except KeyboardInterrupt:
        logger.warning("User interrupted the clarification process")
        print("\nClarification process interrupted.")
        raise
    except Exception as e:
        logger.error(f"Error getting clarification answers: {str(e)}", exc_info=True)
        raise

    logger.info(f"Successfully collected {len(answers)} answers")
    logger.debug(f"Final answers: {answers}")
    return answers

def merge_writeups(
    json_logger: JsonLogger,
    parent_node_id: str,
    run_id: str,
    merge_agent: AgentProvider,
    writeups_to_merge: List[str],
    context: Dict[str, Any]
) -> str:
    """Merge multiple markdown writeups."""
    log_prefix = f"Run {run_id}"
    logger.info(f"{log_prefix}: Starting merge from node {parent_node_id}.")

    node_id = json_logger.start_node(
        parent_node_id=parent_node_id,
        node_type="MERGE_NODE",
        input_data={"writeup_count": len(writeups_to_merge)}
    )

    merged_content = ""
    status = "ERROR"
    error_msg = None
    output_data = {}

    try:
        filtered_writeups = [w for w in writeups_to_merge if w and w.strip()]
        parent_topic = context.get('parent_topic')
        original_task_context_dict = context.get('original_task_context')

        if not filtered_writeups:
            status = "COMPLETED_EMPTY"
            output_data = {"info": "No valid input writeups"}
        elif len(filtered_writeups) == 1:
            merged_content = filtered_writeups[0]
            status = "COMPLETED_SINGLE"
            output_data = {"info": "Single valid input writeup"}
        elif not parent_topic or not original_task_context_dict:
            status = "ERROR"
            error_msg = "Missing required context for merging."
        else:
            logger.info(f"{log_prefix} - Node {node_id}: Merging {len(filtered_writeups)} writeups.")
            merge_prompt = {
                "task": "merge_multiple_content",
                "role": "Expert Technical Editor",
                "instructions": {
                    # Reword goal slightly to use parent_topic and mention context
                    "core_goal": f"Combine the markdown texts provided in 'content_list' into a single, coherent, well-structured, and polished article related to the parent topic ======\n{parent_topic}\n======== and guided by the overall research context provided in 'original_task_context_details'.",
                    # Original structure/formatting instructions
                    "content_and_structure": [
                        "Synthesize the information logically, ensuring smooth transitions between topics derived from different source texts.",
                        "Remove redundant information and repetitive introductory/concluding phrases/sections from the source texts.",
                        "Create a unified introduction that sets the stage for the combined content.",
                        "Create ONLY ONE final concluding section summarizing the key points (unless the context suggests differently).",
                        "DO NOT include a 'Future Research' or similar speculative section unless the context suggests differently).",
                        "Do not put any references in the section or sub-section headings.",
                        "Structure the article logically based on the content. You may reorder sections from the source texts if it improves flow and coherence."
                    ],
                    "formatting_and_references": [
                        "Preserve all markdown formatting (headings, lists, code blocks, tables, etc.). Adjust heading levels as needed for the new structure.",
                        "When merging, pay attention to the section headings and sub-section headings. If they are not consistent, make them consistent.",
                        "Do not create reference numbers, just maintain inline references in markdown format e.g. [title](url).",
                        "Do not put any references other than of the form [title](url) in the text, specifically do not use numbered references like [1], [2], etc.",
                        "Do not create a references section in the end, it will be created in post-processing.",
                        "Ensure consistency in writing style, tone, and terminology throughout."
                    ],
                    "output_format": "Output ONLY the final merged markdown document. Do not include any preamble, commentary, explanations, or apologies."
                },
                "content_list": filtered_writeups,
                # Pass the original context dictionary for reference
                "original_task_context_details": original_task_context_dict
            }    
            prompt_str = json.dumps(merge_prompt, indent=2)
            call_id = json_logger.log_agent_call_start(node_id, "merge_agent", {"input_length": len(prompt_str)})
            agent_response = None
            parsed_output = None
            agent_status = "FAILURE"
            try:
                agent_response = merge_agent.run(prompt_str)
                if isinstance(agent_response, str):
                    merged_content = clean_content_of_markdown_enclosure(agent_response)
                    agent_status = "SUCCESS"
                    status = "COMPLETED"
                    parsed_output = {"merged_length": len(merged_content)}
                    logger.info(f"{log_prefix} - Node {node_id}: Merge completed (length: {len(merged_content)}).")
                else:
                    raise ValueError(f"Merge agent returned unexpected type: {type(agent_response)}")
            except Exception as agent_err:
                error_msg = f"Merge agent call error: {str(agent_err)}"
                logger.error(f"{log_prefix} - Node {node_id}: {error_msg}", exc_info=True)
                status = "ERROR" # Ensure node status is ERROR
            finally:
                json_logger.log_agent_call_end(call_id, agent_status, agent_response, parsed_output, error_msg if agent_status == "FAILURE" else None, merged_content if agent_status == "SUCCESS" else None)

        output_data["final_merged_length"] = len(merged_content)

    except Exception as e:
        error_msg = f"Merge processing error: {str(e)}"
        logger.error(f"{log_prefix} - Node {node_id}: {error_msg}", exc_info=True)
        status = "ERROR"
        output_data = {"final_merged_length": 0}

    finally:
        json_logger.end_node(node_id, status, output_data, error_msg if status == "ERROR" else None)

    return merged_content


def execute_recursive_research(
    json_logger: JsonLogger,
    parent_node_id: Optional[str],
    run_id: str,
    search_ref_agent: AgentProvider,
    structured_agent: AgentProvider,
    merge_agent: AgentProvider,
    task_input: Dict[str, Any],
    key_research_topic: str,
    current_depth: int,
    max_depth: int
) -> Tuple[str, List[Reference]]:
    """Recursive research."""
    log_prefix = f"Run {run_id}: Depth {max_depth - current_depth}/{max_depth}"
    logger.info(f"{log_prefix}: execute_recursive_research started for '{key_research_topic}'")

    if current_depth < 0:
        logger.info(f"{log_prefix}: Depth limit reached for '{key_research_topic}'. Stopping recursion.")
        return "", []

    node_id = json_logger.start_node(
        parent_node_id=parent_node_id,
        node_type="RESEARCH_NODE",
        input_data={ "depth": max_depth - current_depth, "topic": key_research_topic }
    )

    node_writeup = ""
    further_research_topics: List[str] = []
    merged_writeup = ""
    aggregated_references: List[Reference] = []
    status = "ERROR"
    error_msg = None
    search_agent_status = "FAILURE"
    struct_agent_status = "FAILURE"

    try:
        # --- Step 1: Research Current Node Topic ---
        logger.info(f"{log_prefix} - Node {node_id}: Researching '{key_research_topic}'...")
        prompt_dict_search = {
            "task": "research_and_write",
            "instructions": {
                # Original response requirements
                "response_requirements": [
                    "Provide a comprehensive (thorough and detailed) writeup in markdown format for the key_research_topic.",
                ],
                # Modified guidelines to reference the context dict
                "guidelines": [
                    "Pay close attention to the details provided in the original_task_context (initial topic, clarifications, plan summary) to refine the scope, focus, priorities, and exclusions.",
                    "Be informative, accurate, and well-structured.",
                    "Focus only on the key_research_topic for this writeup.", # Keep focus guideline
                    "Generate relevant inline references in markdown format [title](url).",
                    "Do not put any references other than of the form [title](url) in the text, specifically do not use numbered references like [1], [2], etc."
                ]
            },
            "key_research_topic": key_research_topic,
            "original_task_context": task_input, # Pass the full dictionary
        }
        prompt_string_search = json.dumps(prompt_dict_search, indent=2)
        call_id_search = json_logger.log_agent_call_start(node_id, "search_ref_agent", {"topic": key_research_topic})
        search_agent_response, search_parsed_output, search_error = None, None, None
        try:
            search_agent_response = search_ref_agent.run(prompt_string_search)
            if isinstance(search_agent_response, SearchContent):
                node_writeup = clean_content_of_markdown_enclosure(search_agent_response.text)
                search_parsed_output = {"writeup_length": len(node_writeup), "references_count": len(search_agent_response.references)}
                search_agent_status = "SUCCESS"
                aggregated_references.extend([Reference(url=ref.url, title=ref.title) for ref in search_agent_response.references])
                logger.info(f"{log_prefix} - Node {node_id}: Research writeup generated (len: {len(node_writeup)}, refs: {len(search_agent_response.references)}).")
            else: raise ValueError(f"Unexpected search response type: {type(search_agent_response).__name__}")
        except Exception as e_search:
            search_error = f"Search agent error: {str(e_search)}"
            logger.error(f"{log_prefix} - Node {node_id}: {search_error}", exc_info=True)
            if not error_msg: error_msg = search_error
            node_writeup = ""
        finally:
            raw_dump = search_agent_response.model_dump() if isinstance(search_agent_response, SearchContent) else str(search_agent_response)
            json_logger.log_agent_call_end(call_id_search, search_agent_status, raw_dump, search_parsed_output, search_error, node_writeup if search_agent_status == "SUCCESS" else None)

        # --- Step 2: Extract Further Research Topics ---
        if current_depth >= 1 and search_agent_status == "SUCCESS" and node_writeup:
            logger.info(f"{log_prefix} - Node {node_id}: Extracting further topics...")
            extraction_prompt = (
                 f"Analyze the markdown text discussing '{key_research_topic}'. "
                 f"List specific questions or topics mentioned that need further exploration. Focus on knowledge gaps, or next logical sub-topics for research. "
                 f"Focus on 2-4 key areas.\n\n"
                 f"Markdown Text:\n```markdown\n{node_writeup}\n```"
             )
            call_id_struct = json_logger.log_agent_call_start(node_id, "structured_agent_extract", {"input_length": len(extraction_prompt)})
            struct_agent_response, struct_parsed_output, struct_error = None, None, None
            struct_agent_status = "FAILURE"
            try:
                struct_agent_response = structured_agent.run(extraction_prompt, response_schema=FurtherResearch)
                if isinstance(struct_agent_response, FurtherResearch): further_research_topics = struct_agent_response.further_research
                elif isinstance(struct_agent_response, dict) and 'further_research' in struct_agent_response: further_research_topics = struct_agent_response['further_research']
                else: raise ValueError(f"Unexpected struct response type: {type(struct_agent_response)}")

                struct_parsed_output = {"further_research_topics": further_research_topics}
                struct_agent_status = "SUCCESS"
                logger.info(f"{log_prefix} - Node {node_id}: Found {len(further_research_topics)} further topics.")
            except Exception as e_struct:
                struct_error = f"Structured agent error: {str(e_struct)}"
                logger.error(f"{log_prefix} - Node {node_id}: {struct_error}", exc_info=True)
                if not error_msg: error_msg = struct_error
            finally:
                raw_dump = struct_agent_response.model_dump() if isinstance(struct_agent_response, FurtherResearch) else str(struct_agent_response)
                json_logger.log_agent_call_end(call_id_struct, struct_agent_status, raw_dump, struct_parsed_output, struct_error)
        else:
            logger.info(f"{log_prefix} - Node {node_id}: Skipping further topic extraction (Depth {current_depth}, Search Status {search_agent_status}, Writeup {'Exists' if node_writeup else 'Empty'}).")


        # --- Step 3: Recursively Research Children (Sequentially) ---
        child_writeups: List[str] = []
        if current_depth >= 1 and further_research_topics and search_agent_status == "SUCCESS" and struct_agent_status == "SUCCESS":
            logger.info(f"{log_prefix} - Node {node_id}: Starting SEQUENTIAL recursion for {len(further_research_topics)} children...")
            child_context = {"original_task": task_input, "original_topic": key_research_topic} # Pass down context

            for i, child_topic in enumerate(further_research_topics):
                if not child_topic or not isinstance(child_topic, str) or len(child_topic) < 5:
                    logger.warning(f"{log_prefix} - Node {node_id}: Skipping invalid child topic {i+1}: '{child_topic}'")
                    continue

                logger.info(f"{log_prefix} - Node {node_id}: ===> Processing Child {i+1}/{len(further_research_topics)}: '{child_topic}'")
                try:
                    # *** RECURSIVE CALL ***
                    child_writeup, child_refs = execute_recursive_research(
                        json_logger=json_logger, parent_node_id=node_id, run_id=run_id,
                        search_ref_agent=search_ref_agent, structured_agent=structured_agent,
                        merge_agent=merge_agent, task_input=child_context,
                        key_research_topic=child_topic, current_depth=current_depth - 1, max_depth=max_depth
                    )
                    if child_writeup and child_writeup.strip(): child_writeups.append(child_writeup)
                    if child_refs: aggregated_references.extend(child_refs)
                    logger.info(f"{log_prefix} - Node {node_id}: <=== Finished Child {i+1}/{len(further_research_topics)}: '{child_topic}'")
                except Exception as e_child:
                     logger.error(f"{log_prefix} - Node {node_id}: Child task for '{child_topic}' failed: {e_child}", exc_info=True)
                     # Continue processing other children, but log the error
                     if not error_msg: error_msg = f"Error in recursive child '{child_topic[:50]}...': {e_child}"

        # --- Step 4: Merge Writeups ---
        if child_writeups:
            logger.info(f"{log_prefix} - Node {node_id}: Merging node writeup with {len(child_writeups)} child writeup(s).")
            writeups_to_merge = [node_writeup] + child_writeups if node_writeup else child_writeups
            merge_context = {"parent_topic": key_research_topic, "original_task_context": task_input}
            merged_writeup = merge_writeups(
                json_logger=json_logger, parent_node_id=node_id, run_id=run_id,
                merge_agent=merge_agent, writeups_to_merge=writeups_to_merge, context=merge_context
            )
        elif node_writeup:
            logger.info(f"{log_prefix} - Node {node_id}: No children to merge, using node writeup.")
            merged_writeup = node_writeup
        else:
            logger.warning(f"{log_prefix} - Node {node_id}: No content from this node or its children.")
            merged_writeup = ""

        # --- Step 5: Final Reference Deduplication ---
        unique_aggregated_references = deduplicate_references(aggregated_references)
        logger.info(f"{log_prefix} - Node {node_id}: Aggregated {len(unique_aggregated_references)} unique references.")

        # Node status depends only on successful initial search for its own topic
        status = "COMPLETED" if search_agent_status == "SUCCESS" else "ERROR"

    except Exception as e_outer:
        status = "ERROR"
        error_msg = f"Outer research function error: {str(e_outer)}"
        logger.error(f"{log_prefix} - Node {node_id}: Unexpected error: {e_outer}", exc_info=True)
    finally:
        output_data = {
            "final_writeup_length": len(merged_writeup),
            "final_unique_references_count": len(unique_aggregated_references)
        }
        final_node_error = error_msg if status == "ERROR" else None
        json_logger.end_node(node_id, status, output_data, final_node_error)

    logger.info(f"{log_prefix} - Node {node_id}: Finished research for '{key_research_topic}'. Status: {status}")
    # If the initial search failed, return empty results even if children succeeded
    if search_agent_status != "SUCCESS":
         return "", []
    else:
         return merged_writeup, unique_aggregated_references



def run_orchestration(
    run_id: str,
    research_topic: str,
    recursion_depth: int,
    output_base_name: Optional[str],
    skip_clarifications: bool
) -> bool:
    """
    Runs the entire research process in the current process.
    Returns True on success, False on failure.
    """
    json_logger = None
    session_id = None
    final_status = "failed"
    error_message = None

    try:
        # --- Setup Logging and Directories ---
        logger.info(f"--- Starting Run: {run_id} ---")
        run_dir_path = Path(settings.log_directory_path) / run_id
        run_output_dir_path = Path(settings.output_directory_path) / run_id
        run_dir_path.mkdir(parents=True, exist_ok=True)
        run_output_dir_path.mkdir(parents=True, exist_ok=True)

        json_log_file_path = run_dir_path / f"{run_id}.jsonl"
        json_logger = JsonLogger(str(json_log_file_path))

        session_args = { "args": {
                "research_topic_preview": research_topic[:100] + "...",
                "recursion_depth": recursion_depth, "output_base_name": output_base_name,
                "run_id": run_id, "skip_clarifications": skip_clarifications
            }}
        session_id = json_logger.start_session({"args": session_args, "config_preview": {}}) # Simplified config
        logger.info(f"Run {run_id}: Session started ({session_id}). Log: {json_log_file_path}")
        print(f"Run ID: {run_id}. Logging to {json_log_file_path}")

        # --- Initialize Agents ---
        logger.info(f"Run {run_id}: Initializing agents...")
        clarification_agent = create_agent_cli('clarification_questions')
        summary_agent = create_agent_cli('summary')
        search_ref_agent = create_agent_cli('search_ref')
        structured_agent = create_agent_cli('structured_extract')
        merge_agent = create_agent_cli('merge')
        logger.info(f"Run {run_id}: Agents initialized.")

        # --- Clarification Phase ---
        section_input = SectionInput(research_topic=research_topic, recursion_depth=recursion_depth, skip_clarifications=skip_clarifications)
        refined_task_input_dict = section_input.model_dump()

        if not skip_clarifications:
            print("--- Starting Clarification Phase ---")
            clarification_result = run_clarification_phase(
                json_logger=json_logger, parent_node_id=session_id, run_id=run_id,
                clarification_agent=clarification_agent, summary_agent=summary_agent,
                section_input=section_input
            )
            if clarification_result is None:
                raise RuntimeError("Clarification phase failed or was aborted by user.")
            refined_task_input_dict = clarification_result # Update context
            print("--- Clarification Phase Complete ---")
        else:
            logger.info(f"Run {run_id}: Skipping clarification phase.")

        # --- Research Phase ---
        print(f"--- Starting Research (Depth: {recursion_depth}) ---")
        initial_topic = refined_task_input_dict.get("research_topic", "Research")
        max_depth = refined_task_input_dict.get("recursion_depth", 1)

        final_writeup, final_references = execute_recursive_research(
            json_logger=json_logger, parent_node_id=session_id, run_id=run_id,
            search_ref_agent=search_ref_agent, structured_agent=structured_agent,
            merge_agent=merge_agent, task_input=refined_task_input_dict,
            key_research_topic=initial_topic, current_depth=max_depth, max_depth=max_depth
        )
        print("--- Research Phase Complete ---")

        if not final_writeup:
            raise RuntimeError("Research phase failed to generate final writeup.")

        # --- Write Output Files ---
        if not output_base_name:
            output_base_name = run_id
        print(f"--- Writing Output Files (Base: {output_base_name}) ---")
        md_path, html_path = write_output_files(
            final_markdown=final_writeup, output_base_filename=output_base_name,
            output_dir=str(run_output_dir_path), report_title=initial_topic
        )
        if not md_path or not html_path:
            raise RuntimeError("Failed to write output files")
        print(f"Output written to:\n  MD: {md_path}\n  HTML: {html_path}")

        final_status = "completed"
        return True

    except Exception as e:
        final_status = "failed"
        error_message = f"{type(e).__name__}: {e}"
        logger.critical(f"!!! Research run {run_id} failed: {error_message}", exc_info=True)
        print(f"\nERROR: Research run {run_id} failed: {error_message}", file=sys.stderr)
        # Print traceback to console for immediate feedback
        traceback.print_exc(file=sys.stderr)
        return False
    finally:
        if json_logger and session_id:
            try:
                session_status = "COMPLETED" if final_status == "completed" else "ERROR"
                json_logger.session_end(session_id, session_status, {"error": error_message} if error_message else {})
            except Exception as log_err: logger.error(f"Run {run_id}: Failed to log session end: {log_err}")
            json_logger.close()
        logger.info(f"--- Run {run_id} Finished (Status: {final_status}) ---")


# --- CLI Argument Parsing and Main Execution ---
def parse_cli_args():
    """Parses command-line arguments for the CLI runner."""
    parser = argparse.ArgumentParser(description="Scout CLI", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    topic_group = parser.add_mutually_exclusive_group(required=True)
    topic_group.add_argument("--research_topic", help="The research topic as a string.")
    topic_group.add_argument("--research_topic_file", help="Path to a text file containing the research topic.")
    parser.add_argument("--output_name", help="Base name for the output files.")
    parser.add_argument("--recursion_depth", type=int, default=1, help="Recursion depth.")
    parser.add_argument("--skip_clarifications", action="store_true", default=False, help="Skip clarification questions.")
    parser.add_argument("--run_id", default=None, help="Specify a custom run ID.")
    return parser.parse_args()

def main():
    """Main entry point for the CLI."""
    args = parse_cli_args()

    # --- Get Research Topic ---
    research_topic_content = ""
    if args.research_topic:
        research_topic_content = args.research_topic
    elif args.research_topic_file:
        try:
            file_path = Path(args.research_topic_file)
            if not file_path.is_file(): raise FileNotFoundError(f"Topic file not found: {file_path}")
            research_topic_content = file_path.read_text(encoding='utf-8').strip()
            if not research_topic_content: raise ValueError(f"Topic file is empty: {file_path}")
        except Exception as e:
            print(f"ERROR: Could not read research topic file '{args.research_topic_file}': {e}", file=sys.stderr)
            sys.exit(1)

    # --- Prepare Run ID ---
    run_id = args.run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

    start_time = time.monotonic()
    success = run_orchestration(
        run_id=run_id,
        research_topic=research_topic_content,
        recursion_depth=args.recursion_depth,
        output_base_name=args.output_name,
        skip_clarifications=args.skip_clarifications
    )
    duration = time.monotonic() - start_time

    if success:
        print(f"\nResearch completed successfully in {duration:.2f} seconds.")
        sys.exit(0)
    else:
        print(f"\nResearch failed after {duration:.2f} seconds. Check logs in {settings.logging.log_dir}/{run_id}")
        sys.exit(1)

if __name__ == "__main__":
    main()