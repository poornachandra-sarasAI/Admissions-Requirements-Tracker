from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from tavily import TavilyClient


@dataclass(frozen=True)
class UniversitySearchResult:
    """Normalized Tavily search result used for ranking candidate source pages."""

    url: str
    title: str = ""
    content: str = ""
    score: float = 0.0


class AdmissionsCriteria(BaseModel):
    """Admissions criteria extracted from one scraped university page."""

    min_gpa: str | None = Field(
        default=None,
        description="Minimum GPA requirement for the MS Computer Science program, especially for international students.",
    )
    toefl_score: str | None = Field(
        default=None,
        description="TOEFL minimum score, accepted score range, or waiver policy.",
    )
    ielts: str | None = Field(
        default=None,
        description="IELTS minimum score, accepted score range, or waiver policy.",
    )
    gre: str | None = Field(
        default=None,
        description="GRE requirement status, minimum score, optional status, or waiver policy.",
    )
    application_deadline: str | None = Field(
        default=None,
        description="Application deadline for the relevant MS Computer Science intake.",
    )
    application_closes: str | None = Field(
        default=None,
        description="Application closing date if listed separately from the deadline.",
    )
    median_starting_salary: str | None = Field(
        default=None,
        description="Median starting salary for graduates if the page provides it.",
    )
    employment_rate: str | None = Field(
        default=None,
        description="Employment or placement rate for graduates if the page provides it.",
    )
    source_url: str | None = Field(
        default=None,
        description="Source URL from which this page content was scraped.",
    )


def _build_search_queries(university_name: str) -> list[str]:
    """Create the README-planned 3-4 targeted search queries."""

    return [
        f"{university_name} MSCS admission requirements",
        f"{university_name} computer science graduate apply international students",
        f"{university_name} MS computer science GRE TOEFL IELTS GPA",
        f"{university_name} computer science masters application deadline employment salary",
    ]


def _university_tokens(university_name: str) -> set[str]:
    """Return useful tokens for rough official-domain ranking."""

    ignored = {"the", "of", "and", "at", "university", "college", "institute"}
    return {
        token
        for token in university_name.lower().replace("-", " ").split()
        if len(token) > 2 and token not in ignored
    }


def _ranking_score(result: UniversitySearchResult, university_name: str) -> float:
    """Prefer official and admissions-related pages, while preserving Tavily relevance."""

    parsed = urlparse(result.url)
    hostname = parsed.netloc.lower()
    path = parsed.path.lower()
    text = f"{result.title} {result.content} {path}".lower()

    score = float(result.score or 0.0)

    if hostname.endswith(".edu"):
        score += 3.0

    if any(token in hostname for token in _university_tokens(university_name)):
        score += 2.0

    for keyword in ("admission", "graduate", "apply", "computer-science", "computer_science", "cs"):
        if keyword in text:
            score += 0.75

    for keyword in ("toefl", "ielts", "gre", "gpa", "deadline"):
        if keyword in text:
            score += 0.5

    if any(domain in hostname for domain in ("reddit.com", "quora.com", "yocket.com", "collegeconfidential.com")):
        score -= 3.0

    return score


def _dedupe_by_url(results: Iterable[UniversitySearchResult]) -> list[UniversitySearchResult]:
    """Deduplicate results by canonical URL, keeping the first occurrence."""

    seen: set[str] = set()
    unique_results: list[UniversitySearchResult] = []

    for result in results:
        normalized_url = result.url.rstrip("/")
        if normalized_url in seen:
            continue
        seen.add(normalized_url)
        unique_results.append(result)

    return unique_results


def search_university_sites(
    university_name: str,
    *,
    top_k: int = 5,
    results_per_query: int = 5,
    search_depth: str = "advanced",
) -> list[str]:
    """Search the web for likely admissions/source pages for a university.

    This is phase 1 from README.md. It runs several Tavily searches, deduplicates
    URLs, ranks official university/admissions pages higher, and returns the top
    URLs for the later scraping phase.

    Args:
        university_name: User-provided university name, e.g. "Carnegie Mellon University".
        top_k: Number of URLs to return for scraping.
        results_per_query: Tavily results requested per generated query.
        search_depth: Tavily search depth, usually "basic" or "advanced".

    Returns:
        A ranked list of candidate URLs.

    Raises:
        ValueError: If the university name is empty or TAVILY_API_KEY is missing.
    """

    cleaned_name = university_name.strip()
    if not cleaned_name:
        raise ValueError("university_name cannot be empty")

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY is required to search university sites")

    client = TavilyClient(api_key=api_key)
    collected_results: list[UniversitySearchResult] = []

    for query in _build_search_queries(cleaned_name):
        response = client.search(
            query=query,
            max_results=results_per_query,
            search_depth=search_depth,
            include_answer=False,
        )

        for item in response.get("results", []):
            url = item.get("url")
            if not url:
                continue

            collected_results.append(
                UniversitySearchResult(
                    url=url,
                    title=item.get("title", ""),
                    content=item.get("content", ""),
                    score=float(item.get("score") or 0.0),
                )
            )

    unique_results = _dedupe_by_url(collected_results)
    ranked_results = sorted(
        unique_results,
        key=lambda result: _ranking_score(result, cleaned_name),
        reverse=True,
    )

    return [result.url for result in ranked_results[:top_k]]


def _normalize_page_text(text: str) -> str:
    """Collapse noisy whitespace while preserving paragraph boundaries."""

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def scrape_page_content(
    url: str,
    *,
    timeout: int = 20,
    max_chars: int = 30_000,
) -> str:
    """Fetch and clean readable text from a candidate admissions page.

    This is phase 2 from README.md. Call it for the top URLs returned by
    search_university_sites(), then pass the returned text to the extraction
    phase.

    Args:
        url: Candidate source URL to scrape.
        timeout: HTTP request timeout in seconds.
        max_chars: Maximum cleaned text length returned to keep LLM input bounded.

    Returns:
        Cleaned page text prefixed with the source URL.

    Raises:
        ValueError: If the URL is empty or not HTTP(S).
        RuntimeError: If the page cannot be fetched or parsed as text/html.
    """

    cleaned_url = url.strip()
    parsed_url = urlparse(cleaned_url)
    if not cleaned_url or parsed_url.scheme not in {"http", "https"}:
        raise ValueError("url must be a valid http(s) URL")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(cleaned_url, headers=headers, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to fetch {cleaned_url}: {exc}") from exc

    content_type = response.headers.get("content-type", "").lower()
    if "text/html" not in content_type and "application/xhtml" not in content_type:
        raise RuntimeError(f"Unsupported content type for {cleaned_url}: {content_type or 'unknown'}")

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(["script", "style", "noscript", "svg", "iframe", "form"]):
        tag.decompose()

    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    main_content = soup.find("main") or soup.find("article") or soup.body or soup

    for tag in main_content.find_all(["nav", "footer", "header", "aside"]):
        tag.decompose()

    page_text = main_content.get_text("\n", strip=True)
    normalized_text = _normalize_page_text(page_text)

    if not normalized_text:
        raise RuntimeError(f"No readable text found at {cleaned_url}")

    if len(normalized_text) > max_chars:
        normalized_text = normalized_text[:max_chars].rsplit(" ", 1)[0]

    title_line = f"Title: {title}\n" if title else ""
    return f"Source URL: {cleaned_url}\n{title_line}Content:\n{normalized_text}"


def extract_admissions_criteria(
    university_name: str,
    page_content: str,
    *,
    model_name: str = "gpt-4.1-mini",
    max_input_chars: int = 40_000,
) -> AdmissionsCriteria:
    """Extract admissions criteria from one scraped page using structured output.

    This is phase 3 from README.md. Run it once per scraped page, then pass the
    resulting AdmissionsCriteria objects to the later reconcile phase.

    Args:
        university_name: University being researched.
        page_content: Cleaned text returned by scrape_page_content().
        model_name: OpenAI chat model used for extraction.
        max_input_chars: Maximum page text sent to the model.

    Returns:
        AdmissionsCriteria with null values for fields not supported by evidence.

    Raises:
        ValueError: If required inputs or OPENAI_API_KEY are missing.
    """

    cleaned_name = university_name.strip()
    cleaned_content = page_content.strip()

    if not cleaned_name:
        raise ValueError("university_name cannot be empty")
    if not cleaned_content:
        raise ValueError("page_content cannot be empty")
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is required to extract admissions criteria")

    bounded_content = cleaned_content[:max_input_chars]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a careful admissions data extraction assistant. "
                "Extract only facts explicitly supported by the provided page content. "
                "Focus on graduate Master's programs in Computer Science for international/foreign students. "
                "If a field is missing, uncertain, or only available for a different program, return null. "
                "Do not guess or use outside knowledge.",
            ),
            (
                "human",
                "University: {university_name}\n\n"
                "Scraped page content:\n{page_content}",
            ),
        ]
    )

    llm = ChatOpenAI(model=model_name, temperature=0)
    structured_llm = llm.with_structured_output(AdmissionsCriteria)
    chain = prompt | structured_llm

    result = chain.invoke(
        {
            "university_name": cleaned_name,
            "page_content": bounded_content,
        }
    )
    return AdmissionsCriteria.model_validate(result)


ADMISSIONS_FIELDS = [
    "min_gpa",
    "toefl_score",
    "ielts",
    "gre",
    "application_deadline",
    "application_closes",
    "median_starting_salary",
    "employment_rate",
]

EXCEL_COLUMNS = [
    "university_name",
    "min_gpa",
    "toefl_score",
    "ielts",
    "gre",
    "application_deadline",
    "application_closes",
    "median_starting_salary",
    "employment_rate",
    "source_urls",
    "field_sources",
    "field_confidence",
]


def _clean_extracted_value(value: str | None) -> str | None:
    """Normalize extracted values before comparing across pages."""

    if value is None:
        return None

    cleaned = re.sub(r"\s+", " ", str(value)).strip()
    if not cleaned or cleaned.lower() in {"null", "none", "n/a", "na", "not available"}:
        return None
    return cleaned


def _source_priority(source_url: str | None) -> int:
    """Score source URLs so official university pages win ties."""

    if not source_url:
        return 0

    hostname = urlparse(source_url).netloc.lower()
    score = 0
    if hostname.endswith(".edu"):
        score += 3
    if any(domain in hostname for domain in ("reddit.com", "quora.com", "yocket.com", "collegeconfidential.com")):
        score -= 3
    return score


def _confidence_label(value_count: int, total_values: int) -> str:
    """Convert simple agreement counts into an audit-friendly label."""

    if total_values == 0:
        return "missing"
    if value_count >= 2:
        return "high"
    if value_count == 1 and total_values == 1:
        return "medium"
    return "low"


def merge_and_reconcile(
    university_name: str,
    extracted_results: Iterable[AdmissionsCriteria | dict],
) -> dict[str, object]:
    """Merge per-page extraction results into one final admissions row.

    This is phase 4 from README.md. It combines all AdmissionsCriteria objects,
    resolves field-level conflicts, and keeps confidence/source metadata for
    auditability.

    Args:
        university_name: University being researched.
        extracted_results: Per-page outputs from extract_admissions_criteria().

    Returns:
        A flat dictionary ready to be written as one Excel row.
    """

    cleaned_name = university_name.strip()
    if not cleaned_name:
        raise ValueError("university_name cannot be empty")

    normalized_results: list[AdmissionsCriteria] = []
    for result in extracted_results:
        normalized_results.append(AdmissionsCriteria.model_validate(result))

    if not normalized_results:
        raise ValueError("extracted_results must contain at least one item")

    final_row: dict[str, object] = {"university_name": cleaned_name}
    field_sources: dict[str, str | None] = {}
    field_confidence: dict[str, str] = {}

    all_source_urls = [
        result.source_url
        for result in normalized_results
        if _clean_extracted_value(result.source_url)
    ]

    for field_name in ADMISSIONS_FIELDS:
        candidates: list[tuple[str, str | None]] = []

        for result in normalized_results:
            value = _clean_extracted_value(getattr(result, field_name))
            if value:
                candidates.append((value, result.source_url))

        if not candidates:
            final_row[field_name] = None
            field_sources[field_name] = None
            field_confidence[field_name] = "missing"
            continue

        counts = Counter(value for value, _source_url in candidates)
        best_value, best_count = counts.most_common(1)[0]

        tied_values = [value for value, count in counts.items() if count == best_count]
        if len(tied_values) > 1:
            best_value = max(
                tied_values,
                key=lambda value: max(
                    _source_priority(source_url)
                    for candidate_value, source_url in candidates
                    if candidate_value == value
                ),
            )
            best_count = counts[best_value]

        best_sources = [
            source_url
            for value, source_url in candidates
            if value == best_value and _clean_extracted_value(source_url)
        ]
        best_source = max(best_sources, key=_source_priority) if best_sources else None

        final_row[field_name] = best_value
        field_sources[field_name] = best_source
        field_confidence[field_name] = _confidence_label(best_count, len(candidates))

    final_row["source_urls"] = "; ".join(
        dict.fromkeys(url for url in all_source_urls if isinstance(url, str) and url)
    ) or None
    final_row["field_sources"] = field_sources
    final_row["field_confidence"] = field_confidence
    return final_row


def write_to_excel(
    row: dict[str, object],
    *,
    excel_path: str = "college_information.xlsx",
) -> str:
    """Upsert one reconciled admissions row into an Excel file.

    This is phase 5 from README.md. If the Excel file exists, the function
    replaces the matching university row. If it does not exist, it creates a new
    workbook.

    Args:
        row: Final dictionary returned by merge_and_reconcile().
        excel_path: Destination .xlsx file path.

    Returns:
        A short confirmation message.
    """

    if not row.get("university_name"):
        raise ValueError("row must include university_name")

    output_path = Path(excel_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    serializable_row = dict(row)
    for metadata_key in ("field_sources", "field_confidence"):
        value = serializable_row.get(metadata_key)
        if isinstance(value, dict):
            serializable_row[metadata_key] = json.dumps(value)

    new_row_df = pd.DataFrame([serializable_row])

    if output_path.exists() and output_path.stat().st_size > 0:
        existing_df = pd.read_excel(output_path)
    else:
        existing_df = pd.DataFrame(columns=EXCEL_COLUMNS)

    for column in EXCEL_COLUMNS:
        if column not in existing_df.columns:
            existing_df[column] = None
        if column not in new_row_df.columns:
            new_row_df[column] = None

    university_key = str(serializable_row["university_name"]).strip().lower()
    existing_df = existing_df[
        existing_df["university_name"].astype(str).str.strip().str.lower() != university_key
    ]

    combined_df = pd.concat([existing_df, new_row_df[EXCEL_COLUMNS]], ignore_index=True)
    combined_df = combined_df[EXCEL_COLUMNS]
    combined_df.to_excel(output_path, index=False)

    return f"Saved admissions information for {serializable_row['university_name']} to {output_path}"

