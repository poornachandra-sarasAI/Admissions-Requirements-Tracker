# Admissions Requirements Tracker

This project automates collection of MS Computer Science admissions requirements for a university and writes the reconciled results to an Excel file.

It searches the web, scrapes candidate pages, extracts structured fields with an LLM, reconciles conflicts across sources, and persists one row per university.

## What It Captures

- Minimum GPA
- TOEFL score
- IELTS score
- GRE requirement or score
- Application deadline
- Application closes
- Median starting salary
- Employment rate

## Project Files

- `admissions_workflow.py`: CLI entry point for the end-to-end workflow
- `tools.py`: search, scraping, extraction, reconcile, and Excel write logic
- `requirements.txt`: Python dependencies

## Prerequisites

- Python 3.10+
- Tavily API key
- OpenAI API key

## Setup

1. Create and activate a virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Set environment variables (or use a `.env` file).

```bash
export TAVILY_API_KEY="your_tavily_api_key"
export OPENAI_API_KEY="your_openai_api_key"
```

Optional `.env` example:

```env
TAVILY_API_KEY=your_tavily_api_key
OPENAI_API_KEY=your_openai_api_key
```

## Run The Script

Interactive mode:

```bash
python admissions_workflow.py
```

Direct university input:

```bash
python admissions_workflow.py --university "Carnegie Mellon University"
```

Specify output file and tuning options:

```bash
python admissions_workflow.py \
    --university "Carnegie Mellon University" \
    --excel-path college_information.xlsx \
    --top-k 5 \
    --results-per-query 5 \
    --model gpt-4.1-mini
```

## CLI Arguments

- `--university`: University name (if omitted, prompt is shown)
- `--excel-path`: Excel file path (default: `college_information.xlsx`)
- `--top-k`: Number of URLs to scrape (default: `5`)
- `--results-per-query`: Tavily results per search query (default: `5`)
- `--model`: OpenAI model for extraction (default: `gpt-4.1-mini`)

## Output

- Creates or updates an Excel file (default: `college_information.xlsx`)
- Upserts a single reconciled row per university
- Includes source URL metadata for auditability

## Workflow Diagram

User Input: "Carnegie Mellon University"
                 │
                 ▼
┌─────────────────────────────┐
│  1. SEARCH PHASE            │
│  search_university_sites()  │  ◄── Runs 3–4 Tavily queries:
│                             │      "CMU MSCS admission requirements"
│  Returns: [url1, url2, ...] │      "Carnegie Mellon CS graduate apply"
└────────────┬────────────────┘      "CMU MSCS GRE TOEFL GPA"
                         │
                         ▼
┌─────────────────────────────┐
│  2. SCRAPE PHASE            │
│  scrape_page_content()      │  ◄── Called for top 3–5 URLs
│  (called N times)           │      Official site gets priority
│                             │
│  Returns: [text1, text2...] │
└────────────┬────────────────┘
                         │
                         ▼
┌─────────────────────────────┐
│  3. EXTRACTION PHASE        │
│  extract_admissions_        │  ◄── Each page text goes through
│  criteria()                 │      the LLM extractor separately
│  (called N times)           │
│                             │
│  Returns: [dict1, dict2...] │
└────────────┬────────────────┘
                         │
                         ▼
┌─────────────────────────────┐
│  4. RECONCILE PHASE         │
│  merge_and_reconcile()      │  ◄── Combines all dicts, resolves
│                             │      conflicts, marks confidence
│  Returns: final_dict        │      level per field
└────────────┬────────────────┘
                         │
                         ▼
┌─────────────────────────────┐
│  5. PERSIST PHASE           │
│  write_to_excel()           │  ◄── Upserts row into Excel
│                             │      Adds source URLs as metadata
│  Returns: confirmation      │      columns for auditability
└─────────────────────────────┘
