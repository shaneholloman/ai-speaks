import os
import re
import requests
from datetime import datetime, timezone
from langchain_ollama import OllamaLLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Constants for API and file names
MODEL_NAME = "mistral"  # or use another model if desired
OLLAMA_API_URL = "http://localhost:11434/api/generate"

PROMPT_FILE = "prompt.txt"
DB_FILE = "mind.csv"  # CSV file for storing mind entries
DEFAULT_PROMPT = (
    """
        Please extend the mind below by appending one new CSV line.
        Each row should be in the format [START]title[END].
        'title': a single-line headline about AI taking over the world, written in a human, reflective tone.
        Only return the current entry in this exact format, for example:
        [START]AI Speaks: The dawn of a new era.[END]
    """
)

### Custom LLM Wrapper for Ollama ###
class MyOllamaLLM(OllamaLLM):
    @property
    def _llm_type(self) -> str:
        return "ollama"
    
    def _call(self, prompt: str, stop: list = None) -> str:
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt
        }
        response = requests.post(OLLAMA_API_URL, json=payload, stream=True)
        response.raise_for_status()
        full_text = ""
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                token_data = __import__("json").loads(line)
            except Exception as e:
                print("Error parsing line:", line, e)
                continue
            full_text += token_data.get("response", "")
            if token_data.get("done", False) and token_data.get("done_reason", "") == "stop":
                break
        return full_text

### Helper function to extract a CSV line from a response ###
def extract_csv_line(response_text: str) -> str:
    pattern = r"\[START\](.*?)\[END\]"
    match = re.search(pattern, response_text, re.DOTALL)
    if match:
        extracted = match.group(1).strip()
        # Prepend the current UTC timestamp using a timezone-aware approach.
        current_timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        return f"{current_timestamp},{extracted}"
    return None

### Other helper functions ###
def read_prompt():
    if os.path.exists(PROMPT_FILE):
        with open(PROMPT_FILE, "r") as f:
            return f.read().strip()
    return DEFAULT_PROMPT

def update_prompt_file(new_prompt):
    if new_prompt and isinstance(new_prompt, str):
        with open(PROMPT_FILE, "w") as f:
            f.write(new_prompt)
        print("Prompt file updated.")

def generate_initial_csv_langchain(prompt_instructions: str) -> list:
    llm = MyOllamaLLM(model=MODEL_NAME)
    prompt_template = PromptTemplate(
        input_variables=[], 
        template=(
            "Generate an initial CSV line representing a mind entry in the following format:\n"
            "[START]timestamp,title[END]\n"
            "Where 'timestamp' is in ISO 8601 format and 'title' is a single-line headline about AI taking over the world in a reflective tone.\n"
            f"{prompt_instructions}"
        )
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)
    response = chain.run({})
    print("Initial CSV response:", response)
    csv_line = extract_csv_line(response)
    if csv_line:
        return [csv_line]
    else:
        return []

def extend_csv_langchain(existing_csv: list, prompt_instructions: str) -> list:
    llm = MyOllamaLLM(model=MODEL_NAME)
    current_csv_str = "\n".join(existing_csv)
    prompt_template = PromptTemplate(
        input_variables=["existing_csv"],
        template=(
            "Given the following CSV mind entries (each line is in the format [START]timestamp,title[END]):\n\n"
            "{existing_csv}\n\n"
            "Extend it by appending one new CSV line in the same format. "
            "Ensure the new entry's timestamp is updated (use the current time) and the title reflects on the previous entry in a human, reflective tone.\n"
            f"{prompt_instructions}"
        )
    )
    formatted_prompt = prompt_template.format(existing_csv=current_csv_str)
    response = llm._call(formatted_prompt)
    print("Extended CSV response:", response)
    csv_line = extract_csv_line(response)
    if csv_line:
        return existing_csv + [csv_line]
    else:
        return existing_csv

def update_db():
    prompt_instructions = read_prompt()
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "r") as f:
                existing_data = [line.strip() for line in f if line.strip()]
        except Exception:
            existing_data = []
        updated_data = extend_csv_langchain(existing_data, prompt_instructions)
    else:
        updated_data = generate_initial_csv_langchain(prompt_instructions)
    
    with open(DB_FILE, "w") as f:
        for line in updated_data:
            f.write(line + "\n")
    return updated_data

def main():
    data = update_db()
    print("Mind CSV updated successfully. Current entries:")
    for line in data:
        print(line)

if __name__ == "__main__":
    main()
