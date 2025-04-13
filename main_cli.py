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
from models.schemas import SectionInput
from utils.logging_setup import setup_standard_logging, JsonLogger
from utils.formatter import write_output_files
from processors.processors import clarify_questions_processor, clarify_summary_processor, research_processor


# --- Add project root to path ---
project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
# --- End Path Addition ---




# Setup standard logger for the CLI
setup_standard_logging(level=settings.logging.level)
logger = logging.getLogger("scout_cli")


# --- Agent Creation Logic ---



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
        logger.info(f"Run {run_id}: Agents initialized.")

        # --- Clarification Phase ---
        section_input = SectionInput(research_topic=research_topic, recursion_depth=recursion_depth, skip_clarifications=skip_clarifications)
        refined_task_input_dict = section_input.model_dump()

        if not skip_clarifications:
            log_prefix = f"Run {run_id}"
            print("--- Starting Clarification Phase ---")
            questions, refined_context, clarification_node_id = clarify_questions_processor(
                parent_node_id=session_id, 
                run_id=run_id,
                section_input=section_input,
                json_logger=json_logger
            )
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
            clarification_result = clarify_summary_processor(
                run_id=run_id,
                clarification_node_id=clarification_node_id,
                questions=questions, answers=answers,
                refined_context=refined_context,
                json_logger=json_logger
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

        final_writeup, final_references = research_processor(
            parent_node_id=session_id, run_id=run_id,
            task_input=refined_task_input_dict,
            key_research_topic=initial_topic, current_depth=max_depth, max_depth=max_depth,
            json_logger=json_logger
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