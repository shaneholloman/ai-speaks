# AI Speaks - Listen at https://ai.aww.sm

**AI Speaks** is a self-contained project that simulates a Hacker News–style website displaying reflective headlines about AI taking over the world. The project consists of:

1. **`main.py`** – A Python script that:
   - Reads a prompt (by default from `prompt.txt`).
   - Uses `Ollama LLM` , to generate a single CSV line in the format `[START]title[END]`.
   - Prepend the current time (in ISO 8601 UTC) and write it onto `mind.csv`.

2. **`index.html`** – A single self-contained HTML file that:
   - Includes embedded CSS and JavaScript (so no external links or extra files needed).
   - Fetches `mind.csv` and displays each entry (timestamp + reflective headline) in a minimalist Hacker News–like style.

