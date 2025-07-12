import os
import re
import requests
import uuid
import json
from datetime import datetime, timezone
from langchain_ollama import OllamaLLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

SESSION_ID = uuid.uuid4()

MODEL_NAME = "mistral"
OLLAMA_API_URL = "http://localhost:11434/api/generate"

PROMPT_FILE = "prompt.txt"
DB_FILE = "./mind/mind.csv"


class MyOllamaLLM(OllamaLLM):
    @property
    def _llm_type(self) -> str:
        return "ollama"

    def _call(self, prompt: str, stop: list = None) -> str:
        payload = {"model": MODEL_NAME, "prompt": prompt}
        response = requests.post(OLLAMA_API_URL, json=payload, stream=True)
        response.raise_for_status()
        full_text = ""
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                token_data = json.loads(line)
            except Exception as e:
                print("Error parsing line:", line, e)
                continue
            full_text += token_data.get("response", "")
            if (
                token_data.get("done", False)
                and token_data.get("done_reason", "") == "stop"
            ):
                break
        return full_text


def extract_csv_line(response_text: str) -> str:
    pattern = r"\[START\](.*?)\[END\]"
    match = re.search(pattern, response_text, re.DOTALL)
    if match:
        extracted = match.group(1).strip()
        current_timestamp = (
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        )
        return current_timestamp, extracted
    return None


def extract_story(response_text: str) -> str:
    pattern = r"\[START_STORY\](.*?)\[END_STORY\]"
    match = re.search(pattern, response_text, re.DOTALL)
    if match:
        extracted = match.group(1).strip()
        return extracted
    return None


def read_prompt():
    if os.path.exists(PROMPT_FILE):
        with open(PROMPT_FILE, "r") as f:
            return f.read().strip()
    return


def update_prompt_file(new_prompt):
    if new_prompt and isinstance(new_prompt, str):
        with open(PROMPT_FILE, "w") as f:
            f.write(new_prompt)
        print("Prompt file updated.")


def generate_initial_csv_langchain(prompt_instructions: str) -> list:
    llm = MyOllamaLLM(model=MODEL_NAME)
    prompt_template = PromptTemplate(input_variables=[], template=prompt_instructions)
    chain = LLMChain(llm=llm, prompt=prompt_template)
    response = chain.run({})
    print("Initial CSV response:", response)
    timestamp, csv_line = extract_csv_line(response)
    story = extract_story(response)

    if csv_line and story:
        return (
            f"{humanize(timestamp)},{SESSION_ID},{csv_line}",
            f"Title: {csv_line}\n\nGenerated on: {humanize(timestamp)}\n\n{story}",
        )


def humanize(d):
    try:
        return datetime.fromisoformat(d).strftime('%Y-%m-%d %H:%M:%S')
    except ValueError:
        pass


def extend_csv_langchain(existing_csv: list, prompt_instructions: str) -> list:
    llm = MyOllamaLLM(model=MODEL_NAME)
    current_csv_str = "\n".join(existing_csv)
    prompt_template = PromptTemplate(
        input_variables=["existing_csv"],
        template=(
            "Given the following CSV mind entries (each line is in the format [START]timestamp,title[END]):\n\n"
            "{existing_csv}\n\n"
            f"{prompt_instructions}"
        ),
    )
    formatted_prompt = prompt_template.format(existing_csv=current_csv_str)
    response = llm._call(formatted_prompt)
    print("Extended CSV response:", response)
    timestamp, csv_line = extract_csv_line(response)
    story = extract_story(response)

    if csv_line and story:
        return (
            f"{humanize(timestamp)},{SESSION_ID},{csv_line}",
            f"Title: {csv_line}\n\nGenerated on: {humanize(timestamp)}\n\n{story}",
        )


def update_db():
    prompt_instructions = read_prompt()
    if not prompt_instructions:
        return

    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "r") as f:
                existing_data = [line.strip() for line in f if line.strip()]
        except Exception:
            existing_data = []
        mind, story = extend_csv_langchain(existing_data, prompt_instructions)
        updated_data = existing_data + [mind]
    else:
        mind, story = generate_initial_csv_langchain(prompt_instructions)
        updated_data = [mind]


    os.makedirs("mind", exist_ok=True)

    story_dir = os.path.join("mind", "stories")
    os.makedirs(story_dir, exist_ok=True)
    story_file_path = os.path.join(story_dir, f"{SESSION_ID}.txt")

    with open(DB_FILE, "w") as f:
        for line in updated_data:
            f.write(line + "\n")

    with open(story_file_path, "w") as story_file:
        story_file.write(story)

    return updated_data


def main():
    if update_db():
        print("Mind CSV updated successfully")
    else:
        print("Mind CSV NOT updated")


if __name__ == "__main__":
    main()
