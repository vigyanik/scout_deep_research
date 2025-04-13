

import json
import logging

from typing import Any, Dict, Optional, List, Tuple

from config.settings import settings
from models.schemas import SectionInput, Reference, QuestionList, SearchContent, FurtherResearch
from utils.logging_setup import setup_standard_logging, JsonLogger
from utils.helpers import  deduplicate_references, clean_content_of_markdown_enclosure, create_agent_cli

from providers.base_provider import AgentProvider

# Setup standard logger for the CLI
setup_standard_logging(level=settings.logging.level)
logger = logging.getLogger("scout_cli")

def research_processor(
    parent_node_id: Optional[str],
    run_id: str,
    task_input: Dict[str, Any],
    key_research_topic: str,
    current_depth: int,
    max_depth: int,
    search_ref_agent: AgentProvider = None,
    structured_agent: AgentProvider = None,
    merge_agent: AgentProvider = None,
    json_logger: JsonLogger = None
) -> Tuple[str, List[Reference]]:
    if parent_node_id is None:
        logger.error(f"Run {run_id}: Parent node ID is None", exc_info=True)
        return None, None
    if run_id is None:
        logger.error(f"Run {run_id}: Run ID is None", exc_info=True)
        return None, None
    if task_input is None:
        logger.error(f"Run {run_id}: Task input is None", exc_info=True)
        return None, None
    if key_research_topic is None:
        logger.error(f"Run {run_id}: Key research topic is None", exc_info=True)
        return None, None
    if current_depth is None:
        logger.error(f"Run {run_id}: Current depth is None", exc_info=True)
        return None, None
    if max_depth is None:
        logger.error(f"Run {run_id}: Max depth is None", exc_info=True)
        return None, None
        
    if json_logger is None:
        run_dir_path = Path(settings.log_directory_path) / run_id
        json_log_file_path = run_dir_path / f"{run_id}.jsonl"
        json_logger = JsonLogger(str(json_log_file_path))
    if search_ref_agent is None:
        search_ref_agent = create_agent_cli('search_ref')
    if structured_agent is None:
        structured_agent = create_agent_cli('structured_extract')
    if merge_agent is None:
        merge_agent = create_agent_cli('merge')

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
                    child_writeup, child_refs = research_processor(
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
            merged_writeup = merge_processor(
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
         return None, None
    else:
         return merged_writeup, unique_aggregated_references



def clarify_questions_processor(
    parent_node_id: str,
    run_id: str,
    section_input: SectionInput,
    clarification_agent: AgentProvider = None,
    json_logger: JsonLogger = None
) -> Optional[Dict[str, Any]]:
    """
    Clarification phase logic.
    Returns the refined context dictionary or None on failure/abort.
    """
    if parent_node_id is None:
        logger.error(f"Run {run_id}: Parent node ID is None", exc_info=True)
        return None, None, None
    if run_id is None:
        logger.error(f"Run {run_id}: Run ID is None", exc_info=True)
        return None, None, None
    if section_input is None:
        logger.error(f"Run {run_id}: Section input is None", exc_info=True)
        return None, None, None
        
    if json_logger is None:
        run_dir_path = Path(settings.log_directory_path) / run_id
        json_log_file_path = run_dir_path / f"{run_id}.jsonl"
        json_logger = JsonLogger(str(json_log_file_path))
    if clarification_agent is None:
        clarification_agent = create_agent_cli('clarification_questions')
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

    return questions, refined_context, clarification_node_id



def clarify_summary_processor(
    run_id: str,
    clarification_node_id: str,
    questions: List[str],
    answers: List[str],
    refined_context: Dict[str, Any],
    summary_agent: AgentProvider = None,
    json_logger: JsonLogger = None
) -> Optional[Dict[str, Any]]:
    if run_id is None:
        logger.error(f"Run {run_id}: Run ID is None", exc_info=True)
        return None
    if clarification_node_id is None:
        logger.error(f"Run {run_id}: Clarification node ID is None", exc_info=True)
        return None
    if questions is None:
        logger.error(f"Run {run_id}: Questions are None", exc_info=True)
        return None
    if answers is None:
        logger.error(f"Run {run_id}: Answers are None", exc_info=True)
        return None
    if refined_context is None:
        logger.error(f"Run {run_id}: Refined context is None", exc_info=True)
        return None
    if json_logger is None:
        run_dir_path = Path(settings.log_directory_path) / run_id
        json_log_file_path = run_dir_path / f"{run_id}.jsonl"
        json_logger = JsonLogger(str(json_log_file_path))

    if summary_agent is None:
        summary_agent = create_agent_cli('summary')
    log_prefix = f"Run {run_id}"
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



def merge_processor(
    parent_node_id: str,
    run_id: str,
    writeups_to_merge: List[str],
    context: Dict[str, Any],
    merge_agent: AgentProvider = None,
    json_logger: JsonLogger = None
) -> str:
    """Merge multiple markdown writeups."""
    if run_id is None:
        logger.error(f"Run {run_id}: Run ID is None", exc_info=True)
        return None
    if parent_node_id is None:
        logger.error(f"Run {run_id}: Parent node ID is None", exc_info=True)
        return None
    if writeups_to_merge is None:
        logger.error(f"Run {run_id}: Writeups to merge are None", exc_info=True)
        return None
    if context is None:
        logger.error(f"Run {run_id}: Context is None", exc_info=True)
        return None
    if json_logger is None:
        run_dir_path = Path(settings.log_directory_path) / run_id
        json_log_file_path = run_dir_path / f"{run_id}.jsonl"
        json_logger = JsonLogger(str(json_log_file_path))
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


