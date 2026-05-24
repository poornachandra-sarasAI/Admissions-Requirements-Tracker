from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # Optional convenience for local .env files.
    load_dotenv = None

from tools import (
    AdmissionsCriteria,
    extract_admissions_criteria,
    merge_and_reconcile,
    scrape_page_content,
    search_university_sites,
    write_to_excel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Research MS Computer Science admissions requirements and update an Excel file."
    )
    parser.add_argument(
        "--university",
        help="University name. If omitted, the script asks interactively.",
    )
    parser.add_argument(
        "--excel-path",
        default="college_information.xlsx",
        help="Path to the Excel file to create or update.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of search result URLs to scrape.",
    )
    parser.add_argument(
        "--results-per-query",
        type=int,
        default=5,
        help="Number of Tavily results to request for each generated query.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="OpenAI model used for structured extraction.",
    )
    return parser.parse_args()


def get_university_name(cli_value: str | None) -> str:
    if cli_value and cli_value.strip():
        return cli_value.strip()

    university_name = input("Enter the university name: ").strip()
    if not university_name:
        raise ValueError("University name cannot be empty.")
    return university_name


def collect_page_extractions(
    university_name: str,
    urls: list[str],
    *,
    model_name: str,
) -> list[AdmissionsCriteria]:
    extractions: list[AdmissionsCriteria] = []

    for index, url in enumerate(urls, start=1):
        print(f"[{index}/{len(urls)}] Scraping {url}")

        try:
            page_content = scrape_page_content(url)
        except Exception as exc:  # Keep the workflow moving when one page fails.
            print(f"    Skipped scrape: {exc}")
            continue

        try:
            criteria = extract_admissions_criteria(
                university_name,
                page_content,
                model_name=model_name,
            )
        except Exception as exc:
            print(f"    Skipped extraction: {exc}")
            continue

        if not criteria.source_url:
            criteria = criteria.model_copy(update={"source_url": url})

        extractions.append(criteria)
        found_fields = [
            field
            for field, value in criteria.model_dump().items()
            if field != "source_url" and value
        ]
        print(f"    Extracted fields: {', '.join(found_fields) if found_fields else 'none'}")

    return extractions


def run_admissions_workflow(
    university_name: str,
    *,
    excel_path: str = "college_information.xlsx",
    top_k: int = 5,
    results_per_query: int = 5,
    model_name: str = "gpt-4.1-mini",
) -> dict[str, object]:
    print(f"Searching for admissions pages for {university_name}...")
    urls = search_university_sites(
        university_name,
        top_k=top_k,
        results_per_query=results_per_query,
    )

    if not urls:
        raise RuntimeError(f"No candidate URLs found for {university_name}.")

    print("Candidate URLs:")
    for url in urls:
        print(f"  - {url}")

    extractions = collect_page_extractions(
        university_name,
        urls,
        model_name=model_name,
    )

    if not extractions:
        raise RuntimeError("No admissions criteria could be extracted from the candidate pages.")

    print("Reconciling extracted fields...")
    final_row = merge_and_reconcile(university_name, extractions)

    print(f"Writing results to {excel_path}...")
    confirmation = write_to_excel(final_row, excel_path=excel_path)
    print(confirmation)

    return final_row


def main() -> int:
    if load_dotenv is not None:
        load_dotenv()

    args = parse_args()

    try:
        university_name = get_university_name(args.university)
        final_row = run_admissions_workflow(
            university_name,
            excel_path=args.excel_path,
            top_k=args.top_k,
            results_per_query=args.results_per_query,
            model_name=args.model,
        )
    except Exception as exc:
        print(f"Workflow failed: {exc}", file=sys.stderr)
        return 1

    print("\nFinal reconciled row:")
    for key, value in final_row.items():
        print(f"{key}: {value}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
