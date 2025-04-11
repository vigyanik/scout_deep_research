# Scout Deep Research

## Overview

Scout is a Python-based system designed to automate the process of conducting research and generating reports. It leverages AI agents across multiple different providers, to gather information from various sources and compile it into structured outputs. The system is designed to be modular and extensible, supporting different agent providers (e.g., Gemini, OpenAI) and output formats.

## Key Features

* **Modular Design:** Scout is built with a modular architecture, allowing for easy integration of different components.
* **Agent-Based Research:** The system uses AI agents to perform research tasks. These agents are implemented in the `providers` directory and can be configured to use different large language models (LLMs).
* **Configurable Workflow:** The research process is driven by a JSON configuration file (`config/config.json`), which specifies the agents to use, API keys, and other settings.
* **Output Generation:** Scout generates reports in Markdown and HTML formats.
* **Logging:** The system uses a JSON Lines format for logging, providing structured and detailed information about the research process.
* **Command-Line Interface (CLI):** The `main_cli.py` script provides a command-line interface to run research tasks.


## Usage

1. **Clone the repo**
```bash
git clone https://github.com/vigyanik/scout_deep_research
cd scout_deep_research
```
2. **Install Dependencies:**
```bash
pip install -r requirements.txt
```
3. **API Keys:** Add your API keys for the desired LLM providers (e.g., OpenAI, Gemini) in config.json
```bash
cd config
cp config_minimal.json config.json
```
Edit config.json and add your keys and configure models.
```json
{
    "api_keys": {
    "gemini": "YOUR-GEMINI-KEY"
    },
    ...
}
```
Then navigate back to the main directory
```bash
cd ..
```
4. **Run**
```bash
python main_cli.py --research_topic "Your detailed research topic or question here" 
```
OR
```bash
python main_cli.py --research_topic_file path/to/your/topic_file.txt
```
by default, it will prompt you with clarifying questions.
outputs are produced in HTML (.html) and Markdown (.md) formats
```bash
ls output_reports/run_<id>/
```
## Project Structure

The project is organized into the following main directories:

* `config/`: Contains configuration files (`config.json`, `settings.py`).
* `models/`: Defines Pydantic schemas for data structures used throughout the system.
* `providers/`: Implements the agents that interact with LLMs and other data sources (e.g., Gemini, OpenAI).
* `utils/`: Provides utility functions for logging, text processing, and other common tasks.

## Modules

### Configuration

The `config` directory provides a structured way to manage settings:

* `settings.py`: Defines the `Settings` class, which loads configuration from `config.json` and environment variables using Pydantic. It handles API keys, model configurations, agent settings, logging, and output directories.
* `config.json`: Stores the main configuration data, including API keys and model settings.

### Data Models

The `models` directory defines the structure of data used within the application:

* `schemas.py`:  Contains Pydantic models representing various data entities, including `SearchReference`, `SearchContent`, `SectionInput`, `FurtherResearch`, `MergedWriteup`, `QuestionList`, `StatusUpdate`, and `LogUpdateMessage`.

### Providers

The `providers` directory contains implementations for different agent providers:

* `base_provider.py`: Defines the abstract `AgentProvider` class, which serves as the base class for all specific agent providers.
* `gemini_provider.py` and `gemini_provider_v2.py`: Implement agents using the Gemini LLM, including agents for search, structured data extraction, and text merging.
* `openai_provider.py` and `openai_provider_v2.py`: Implement agents using the OpenAI LLM, with similar functionalities.

### Utilities

The `utils` directory provides various helper functions:

* `logging_setup.py`: Configures standard Python logging and sets up a custom `JsonLogger` for structured logging.
* `helpers.py`:  Includes utility functions for tasks like deduplicating references, cleaning Markdown content, and rate limiting.
* `formatter.py`:  Handles Markdown to HTML conversion, including table of contents generation and reference formatting.


### Main Script

* `main_cli.py`: A command-line interface designed to run the research process from the terminal.
