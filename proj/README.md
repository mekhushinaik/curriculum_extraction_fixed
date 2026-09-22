# Agentic AI Curriculum Extraction

Universal textbook pipeline for CBSE Grade 10 (and other grades/subjects).
The same code works for **Mathematics, Science, Social Science, or any other textbook**; the subject is a command-line input, not hard-coded.

## Architecture

PDF → **Docling parser agent** → chapter chunks → **LLM structuring/reviewer agent** → Pydantic validation → retry/self-correction → final JSON

This uses **LangGraph** for orchestration and **Instructor + Pydantic** for typed structured output. Docling converts PDFs into layout-aware Markdown. citeturn0search0turn0search5

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:GROQ_API_KEY="YOUR_GROQ_KEY"
```

If PowerShell blocks activation, use:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run any textbook

```powershell
python main.py --pdf "CBSE_Science.pdf" --subject "Science" --grade "10" --curriculum "CBSE"
python main.py --pdf "CBSE_Social_Science.pdf" --subject "Social Science" --grade "10" --curriculum "CBSE"
```

Output files are named automatically, for example:
- `output_science.json`
- `output_social-science.json`

## ID rules

Textbook heading/TOC → `-tb-`
AI-inferred grouping → `-ai-`

Examples:
`top-social-science-tb-1-1`
`sub-social-science-ai-1-1-1`

## Important fix from the earlier version

The old vision strategy treated **each PDF page as a chapter**, which is incorrect. For the two-hour submission, use the Docling pipeline for the full textbook. If you want the second ingestion comparison, compare Docling with a vision parser on a representative chapter rather than pretending each page is a chapter.

## Deliverables checklist

1. Framework setup: LangGraph in `graph.py`.
2. Parser agent: Docling in `parsers.py`.
3. Structuring/reviewer agent: Instructor + Pydantic in `graph.py`.
4. Validation loop: `validate_node` → retry → `structure_node`.
5. Universal subject handling: `--subject` CLI argument.
6. Final JSON: `output_<subject>.json`.
7. Final report: record runtime, extraction quality, and `-tb-`/`-ai-` accuracy after one real run.

## Current Groq note

For vision experiments, use a currently supported multimodal model such as `meta-llama/llama-4-maverick-17b-128e-instruct`; the older `llama-3.2-90b-vision-preview` in the original ZIP is no longer the right model choice. Groq's current docs list Llama 4 vision models and Qwen multimodal models. citeturn0search2turn0search4
