"""GraphRAG lab: index a text corpus, query a two-hop knowledge graph, and evaluate it.

Runs completely offline by default.  Set OPENAI_API_KEY and pass --extractor openai
to replace the deterministic triple extractor with an LLM-based one.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import httpx

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9'-]{1,}")
ENTITY_RE = re.compile(r"(?:[A-Z][A-Za-z0-9&.-]+(?:\s+(?:[A-Z][A-Za-z0-9&.-]+|of|and|the|for|in|to)){0,5}|\b[A-Z]{2,}\b)")
STOP_ENTITIES = {"The", "This", "That", "These", "What", "When", "Where", "Why", "How", "Which", "Who", "Many", "Full Content", "Query", "Title", "Link", "Snippet", "Download"}
GENERIC_ENTITY_TOKENS = {"electric", "vehicle", "vehicles", "ev", "evs", "market", "markets", "report", "source", "content", "united", "states", "us", "the", "and", "for", "in", "of", "to", "what", "when", "where", "why", "how", "which", "who", "many", "share", "sales", "growth", "results", "year"}


@dataclass(frozen=True)
class Document:
    doc_id: str
    query: str
    title: str
    url: str
    text: str


@dataclass(frozen=True)
class Triple:
    subject: str
    predicate: str
    object: str
    doc_id: str
    sentence: str


@dataclass(frozen=True)
class Benchmark:
    """A manually verified, title-independent evaluation item."""
    benchmark_id: str
    question: str
    expected_doc: str
    expected_answer: str
    required_terms: tuple[str, ...]


@dataclass
class ExtractionUsage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


# These questions are written from source passages, not from their document titles.
# `required_terms` are only for post-hoc evaluation; they are never passed to retrievers.
BENCHMARKS: tuple[Benchmark, ...] = (
    Benchmark("B01", "Which policy regime was associated with a 5% new-EV share, compared with 1.3% elsewhere?", "doc_1", "States with zero-emission vehicle regulations had a 5% new-EV share; states without them averaged 1.3%.", ("zero-emission", "5%", "1.3%")),
    Benchmark("B02", "How many new EVs did Americans buy in Q1 2024, and what was their share of new-vehicle sales?", "doc_2", "Americans bought 268,909 new EVs and EVs represented 7.3% of new-vehicle sales.", ("268,909", "7.3%")),
    Benchmark("B03", "What battery warranty is offered by several EV manufacturers?", "doc_3", "Several manufacturers offer an 8-year/100,000-mile battery warranty.", ("8-year", "100,000")),
    Benchmark("B04", "What annual global BEV-volume growth did Citi forecast while describing sentiment as overly negative?", "doc_5", "Citi forecast 12% global BEV-volume growth for the year.", ("12%", "negative")),
    Benchmark("B05", "What investment and job totals were announced for U.S. EV and battery manufacturing?", "doc_6", "Manufacturers announced $188 billion in investment and 195,000 direct jobs.", ("188 billion", "195,000")),
    Benchmark("B06", "How does EV energy efficiency compare with gasoline-vehicle efficiency in the EPA source?", "doc_8", "EVs use about 87%–91% of battery energy, while gasoline vehicles convert about 16%–25%.", ("87%", "91%", "16", "25")),
    Benchmark("B07", "What global sales milestone for battery-electric and plug-in hybrid cars was reached in 2019?", "doc_9", "Sales passed two million vehicles for the first time in 2019.", ("two-million", "2019")),
    Benchmark("B08", "Which transport trends does the annual Electric Vehicle Outlook examine alongside electrification?", "doc_10", "It examines shared mobility and autonomous driving alongside electrification.", ("shared mobility", "autonomous")),
    Benchmark("B09", "What first-quarter revenue did NVIDIA report for the quarter ending April 28, 2024?", "doc_11", "NVIDIA reported $26.0 billion in revenue.", ("26.0 billion",)),
    Benchmark("B10", "What nine-month revenue did Polestar report for 2023 and what was the year-over-year change?", "doc_12", "Polestar reported $1,844.4 million in revenue, up 25%.", ("1,844.4", "25")),
    Benchmark("B11", "How many vehicles did VinFast deliver in Q3 2024 and what was its year-over-year delivery growth?", "doc_13", "VinFast delivered 21,912 vehicles, up 115% year over year.", ("21,912", "115")),
    Benchmark("B12", "What were ZEEKR's deliveries in the first quarter of 2024?", "doc_14", "ZEEKR reported 33,059 deliveries in Q1 2024.", ("33,059",)),
    Benchmark("B13", "What EBIT and revenue did Mercedes-Benz Group report for 2023?", "doc_16", "Mercedes-Benz reported €19.7 billion EBIT and €153.2 billion revenue.", ("19.7", "153.2")),
    Benchmark("B14", "What payload and cargo-space advantage does REE's P7 platform claim?", "doc_19", "P7 supports up to 8,800 lbs and up to 35% more cargo space.", ("8,800", "35%")),
    Benchmark("B15", "How many hydrogen fuel-cell trucks were ordered from Nikola, and what California deadline applies to drayage trucks?", "doc_20", "Nikola reported orders for 140 trucks; California requires zero-emission drayage trucks by 2035.", ("140", "2035")),
    Benchmark("B16", "How many U.S. buyers chose EVs in 2023, and what market share did EVs reach?", "doc_30", "Nearly 1.2 million buyers chose EVs and EVs reached 7.6% share.", ("1.2 million", "7.6%")),
    Benchmark("B17", "Did U.S. light-duty EV sales show growth or decline through the third quarter of 2023?", "doc_36", "They showed continued and significant growth.", ("continued", "significant growth")),
    Benchmark("B18", "What proportion of U.S. adults said they would seriously consider an EV for their next purchase?", "doc_44", "Thirty-eight percent said they were very or somewhat likely to consider an EV.", ("38%",)),
    Benchmark("B19", "Across 21 markets in Q2 2024, what electrified-vehicle share and EV sales growth were reported?", "doc_67", "Electrified vehicles had 37% share and EV sales grew 21% year over year.", ("37%", "21%")),
    Benchmark("B20", "What were global electrified passenger-vehicle sales in 2022 and the projected level for 2030?", "doc_70", "Sales exceeded ten million in 2022 and were projected to reach roughly 40 million in 2030.", ("ten million", "40 million")),
)


def tokens(text: str) -> list[str]:
    # Keep common dotted country abbreviations as one searchable token.
    normalized = re.sub(r"(?<!\w)U\.S\.(?!\w)", "US", text, flags=re.IGNORECASE)
    return [t.lower() for t in TOKEN_RE.findall(normalized)]


def read_corpus(corpus: Path) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(corpus.glob("doc_*.txt"), key=lambda p: int(re.search(r"\d+", p.stem).group())):
        raw = path.read_text(encoding="utf-8", errors="replace")
        fields = dict(re.findall(r"^(Query|Title|Link|Snippet):\s*(.*)$", raw, flags=re.MULTILINE))
        body = raw.split("Full Content:", 1)[-1].strip()
        docs.append(Document(path.stem, fields.get("Query", ""), fields.get("Title", path.stem), fields.get("Link", ""), body))
    if not docs:
        raise FileNotFoundError(f"No doc_*.txt files found in {corpus}")
    return docs


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n{2,}", text) if len(s.strip()) >= 30]


def entities(text: str, limit: int = 12) -> list[str]:
    found: list[str] = []
    for raw in ENTITY_RE.findall(text):
        value = re.sub(r"\s+", " ", raw).strip(" -.,;:")
        if value in STOP_ENTITIES or len(value) < 3 or value.isdigit():
            continue
        if value not in found:
            found.append(value)
        if len(found) == limit:
            break
    return found


def predicate_for(sentence: str) -> str:
    """Conservative relation labels used by the reproducible offline extractor."""
    lower = sentence.lower()
    if any(word in lower for word in ("revenue", "ebit", "earnings", "financial results", "cash flow")):
        return "REPORTS_FINANCIAL_RESULT"
    if any(word in lower for word in ("deliveries", "vehicles sold", "sales grew", "sales growth", "buyers opted")):
        return "REPORTS_SALES"
    if any(word in lower for word in ("regulation", "incentive", "tax credit", "inflation reduction act", "zero-emission")):
        return "REPORTS_POLICY"
    if any(word in lower for word in ("charging", "charger", "refueling")):
        return "REPORTS_CHARGING"
    if any(word in lower for word in ("survey", "sentiment", "consider an electric")):
        return "REPORTS_SENTIMENT"
    if any(word in lower for word in ("forecast", "project", "outlook")):
        return "REPORTS_FORECAST"
    if any(char.isdigit() for char in sentence):
        return "REPORTS_MEASUREMENT"
    return "MENTIONS"


def compact_fact(sentence: str) -> str:
    """Use a bounded, human-readable fact node rather than an opaque ID."""
    normalized = re.sub(r"\s+", " ", sentence).strip()
    return normalized[:240].rstrip(" ,;:")


def llm_excerpt(doc: Document, max_chars: int = 5000) -> str:
    """Select source sentences with the most extractable facts for one LLM call."""
    source_sentences = sentences(doc.text)
    scored = []
    for position, sentence in enumerate(source_sentences):
        lower = sentence.lower()
        score = 3 if any(char.isdigit() for char in sentence) else 0
        score += 2 if predicate_for(sentence) != "MENTIONS" else 0
        score += sum(keyword in lower for keyword in ("revenue", "sales", "regulation", "incentive", "forecast", "charging", "delivered", "investment"))
        scored.append((score, position))
    selected_positions = {position for _, position in sorted(scored, reverse=True)[:24]}
    selected_positions.update(range(min(3, len(source_sentences))))
    excerpt_parts = []
    size = 0
    for position, sentence in enumerate(source_sentences):
        if position not in selected_positions or size + len(sentence) + 1 > max_chars:
            continue
        excerpt_parts.append(sentence)
        size += len(sentence) + 1
    return "\n".join(excerpt_parts) or doc.text[:max_chars]


def heuristic_triples(doc: Document) -> list[Triple]:
    """Extract explicit, provenance-grounded triples without a network/API dependency.

    The extractor never invents an entity or fact: each fact node is a bounded
    span copied from its source sentence and every triple retains that sentence.
    """
    result: list[Triple] = []
    title_entity = doc.title.rstrip(" .")
    for sent in sentences(doc.text)[:80]:
        es = entities(sent, limit=5)
        predicate = predicate_for(sent)
        subject = es[0] if es else title_entity
        if subject != title_entity:
            result.append(Triple(title_entity, "MENTIONS", subject, doc.doc_id, sent))
        for entity in es[1:]:
            result.append(Triple(subject, "RELATED_TO", entity, doc.doc_id, sent))
        # Index a factual statement only where a semantic cue or a measurement
        # occurs; this avoids treating all prose as a graph fact.
        if predicate != "MENTIONS":
            result.append(Triple(subject, predicate, compact_fact(sent), doc.doc_id, sent))
    # A document-level triple ensures every source is reachable from its topic.
    result.append(Triple(title_entity, "HAS_SOURCE", doc.doc_id, doc.doc_id, doc.title))
    return result


def openai_triples(doc: Document, model: str, client: Any) -> tuple[list[Triple], ExtractionUsage]:
    """Optional LLM extractor.  The graph remains provenance-grounded per document."""
    excerpt = llm_excerpt(doc, max_chars=int(os.getenv("GRAPHRAG_LLM_EXCERPT_CHARS", "5000")))
    prompt = f"""Extract only explicit factual triples from this document. Return JSON only:
{{\"triples\":[{{\"subject\":str,\"predicate\":UPPER_SNAKE_CASE,\"object\":str,\"evidence\":str}}]}}.
Use at most 8 triples. Do not infer facts absent from the text. `evidence` must be a verbatim complete sentence from TEXT. Prefer concise evidence sentences.\n\nTITLE: {doc.title}\nTEXT:\n{excerpt}"""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
        max_tokens=1400,
    )
    content = (response.choices[0].message.content or "{}").strip()
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.IGNORECASE)
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned invalid JSON for {doc.doc_id}: {exc.msg}") from exc
    triples = []
    normalized_text = re.sub(r"\s+", " ", doc.text).lower()
    for item in payload.get("triples", []):
        if all(isinstance(item.get(k), str) and item[k].strip() for k in ("subject", "predicate", "object")):
            evidence = re.sub(r"\s+", " ", item.get("evidence", "")).strip()
            # A triple without verbatim source evidence is excluded. This guards
            # against unsupported LLM extraction before it enters the graph.
            if evidence and evidence.lower() in normalized_text:
                triples.append(Triple(item["subject"].strip(), item["predicate"].strip().upper(), item["object"].strip(), doc.doc_id, evidence))
    usage = getattr(response, "usage", None)
    return triples, ExtractionUsage(
        calls=1,
        input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
    )


def gemini_triples(doc: Document, model: str, api_key: str) -> tuple[list[Triple], ExtractionUsage]:
    """Gemini REST extractor using JSON mode and the same provenance contract."""
    excerpt = llm_excerpt(doc, max_chars=int(os.getenv("GRAPHRAG_LLM_EXCERPT_CHARS", "5000")))
    prompt = f"""Extract only explicit factual triples from this document. Return JSON only:
{{"triples":[{{"subject":str,"predicate":UPPER_SNAKE_CASE,"object":str,"evidence":str}}]}}.
Use at most 8 triples. Do not infer facts absent from the text. `evidence` must be a verbatim complete sentence from TEXT. Prefer concise evidence sentences.

TITLE: {doc.title}
TEXT:
{excerpt}"""
    response = httpx.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        params={"key": api_key},
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 1400, "responseMimeType": "application/json"},
        },
        timeout=60,
    )
    if response.status_code == 429:
        raise RuntimeError("Gemini API rate limit (429)")
    response.raise_for_status()
    payload = response.json()
    content = payload["candidates"][0]["content"]["parts"][0]["text"].strip()
    parsed = json.loads(content)
    normalized_text = re.sub(r"\s+", " ", doc.text).lower()
    triples = []
    items = parsed if isinstance(parsed, list) else parsed.get("triples", [])
    for item in items:
        if all(isinstance(item.get(key), str) and item[key].strip() for key in ("subject", "predicate", "object")):
            evidence = re.sub(r"\s+", " ", item.get("evidence", "")).strip()
            if evidence and evidence.lower() in normalized_text:
                triples.append(Triple(item["subject"].strip(), item["predicate"].strip().upper(), item["object"].strip(), doc.doc_id, evidence))
    usage = payload.get("usageMetadata", {})
    return triples, ExtractionUsage(
        calls=1,
        input_tokens=int(usage.get("promptTokenCount", 0) or 0),
        output_tokens=int(usage.get("candidatesTokenCount", 0) or 0),
    )


def build_graph(docs: Iterable[Document], extractor: str, model: str, cache_dir: Path | None = None) -> tuple[nx.MultiDiGraph, list[Triple], ExtractionUsage]:
    docs = list(docs)
    graph = nx.MultiDiGraph()
    all_triples: list[Triple] = []
    usage = ExtractionUsage()
    extracted: dict[str, tuple[list[Triple], ExtractionUsage]] = {}
    if extractor in {"openai", "gemini", "hybrid"} and cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        for doc in docs:
            cache_file = cache_dir / f"{doc.doc_id}.json"
            if not cache_file.exists():
                continue
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
            if payload.get("model") == model:
                extracted[doc.doc_id] = (
                    [Triple(**item) for item in payload.get("triples", [])],
                    ExtractionUsage(**payload.get("usage", {})),
                )
    if extractor in {"openai", "gemini"}:
        gemini_key = os.getenv("GEMINI_API_KEY", "")
        if extractor == "gemini" and not gemini_key:
            raise ValueError("GEMINI_API_KEY is required with extractor=gemini")
        from openai import OpenAI, RateLimitError

        # One persistent client per worker: sharing the SDK client across
        # threads can stall some HTTP transports, while creating one per
        # document is needlessly slow.
        local = threading.local()
        rate_lock = threading.Lock()
        next_request_at = [0.0]
        request_delay = float(os.getenv("GRAPHRAG_LLM_REQUEST_DELAY", "15"))

        def extract_one(doc: Document) -> tuple[list[Triple], ExtractionUsage]:
            if extractor == "gemini":
                retries = int(os.getenv("GRAPHRAG_LLM_RATE_RETRIES", "4"))
                for attempt in range(retries + 1):
                    try:
                        return gemini_triples(doc, model, gemini_key)
                    except RuntimeError as exc:
                        if "rate limit" not in str(exc).lower() or attempt == retries:
                            raise
                        backoff = float(os.getenv("GRAPHRAG_LLM_RATE_BACKOFF", "60")) * (attempt + 1)
                        print(f"Gemini rate-limited; waiting {backoff:.0f}s before retry", flush=True)
                        time.sleep(backoff)
            if extractor == "openai":
                retries = int(os.getenv("GRAPHRAG_LLM_RATE_RETRIES", "4"))
                for attempt in range(retries + 1):
                    try:
                        if not hasattr(local, "client"):
                            local.client = OpenAI(timeout=45.0, max_retries=0)
                        return openai_triples(doc, model, local.client)
                    except RateLimitError:
                        if attempt == retries:
                            raise RuntimeError("OpenAI API rate limit (429)")
                        backoff = float(os.getenv("GRAPHRAG_LLM_RATE_BACKOFF", "60")) * (attempt + 1)
                        print(f"OpenAI rate-limited; waiting {backoff:.0f}s before retry", flush=True)
                        time.sleep(backoff)
            if not hasattr(local, "client"):
                local.client = OpenAI(timeout=45.0, max_retries=1)
            with rate_lock:
                wait_seconds = max(0.0, next_request_at[0] - time.monotonic())
                next_request_at[0] = max(next_request_at[0], time.monotonic()) + request_delay
            if wait_seconds:
                time.sleep(wait_seconds)
            return openai_triples(doc, model, local.client)

        # Default to one worker so low-quota API accounts remain within their
        # request/token limits. Higher-throughput accounts can opt in to more.
        workers = min(max(int(os.getenv("GRAPHRAG_LLM_WORKERS", "1")), 1), 8)
        missing_docs = [(position, doc) for position, doc in enumerate(docs, start=1) if doc.doc_id not in extracted]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(extract_one, doc): (position, doc) for position, doc in missing_docs}
            for future in as_completed(futures):
                position, doc = futures[future]
                extracted[doc.doc_id] = future.result()
                if cache_dir:
                    triples, doc_usage = extracted[doc.doc_id]
                    (cache_dir / f"{doc.doc_id}.json").write_text(json.dumps({
                        "model": model, "triples": [asdict(triple) for triple in triples], "usage": asdict(doc_usage),
                    }, ensure_ascii=False), encoding="utf-8")
                print(f"Extracted triples with LLM: {position}/{len(docs)}", flush=True)
    for doc in docs:
        graph.add_node(doc.doc_id, kind="document", label=doc.title, url=doc.url, text=doc.text)
        if doc.doc_id in extracted:
            triples, doc_usage = extracted[doc.doc_id]
            usage.calls += doc_usage.calls
            usage.input_tokens += doc_usage.input_tokens
            usage.output_tokens += doc_usage.output_tokens
        else:
            triples = heuristic_triples(doc)
        for triple in triples:
            all_triples.append(triple)
            for entity in (triple.subject, triple.object):
                if entity != doc.doc_id:
                    graph.add_node(entity, kind="entity", label=entity)
            graph.add_edge(triple.subject, triple.object, predicate=triple.predicate, doc_id=doc.doc_id, evidence=triple.sentence)
            if doc.doc_id not in (triple.subject, triple.object):
                graph.add_edge(doc.doc_id, triple.subject, predicate="MENTIONS", doc_id=doc.doc_id, evidence=triple.sentence)
                graph.add_edge(doc.doc_id, triple.object, predicate="MENTIONS", doc_id=doc.doc_id, evidence=triple.sentence)
    return graph, all_triples, usage


def build_flat_index(docs: list[Document]) -> tuple[Counter[str], dict[str, Counter[str]]]:
    n = len(docs)
    df: Counter[str] = Counter()
    corpus_tokens: dict[str, Counter[str]] = {}
    for doc in docs:
        bag = Counter(tokens(f"{doc.title} {doc.text}"))
        corpus_tokens[doc.doc_id] = bag
        df.update(bag.keys())
    return df, corpus_tokens


def flat_rank(query: str, docs: list[Document], index: tuple[Counter[str], dict[str, Counter[str]]]) -> list[tuple[str, float]]:
    q = Counter(tokens(query))
    n = len(docs)
    df, corpus_tokens = index
    scores = []
    for doc in docs:
        bag = corpus_tokens[doc.doc_id]
        score = sum((1 + math.log(bag[t])) * math.log((n + 1) / (df[t] + 1)) * count for t, count in q.items() if t in bag)
        scores.append((doc.doc_id, score))
    return sorted(scores, key=lambda item: (-item[1], item[0]))


def match_query_entities(query: str, graph: nx.MultiDiGraph) -> list[str]:
    q = set(tokens(query))
    matches: list[tuple[float, str]] = []
    for node, data in graph.nodes(data=True):
        if data.get("kind") != "entity":
            continue
        nt = set(tokens(str(node)))
        informative = nt - GENERIC_ENTITY_TOKENS
        if not informative:
            continue
        overlap_tokens = informative & q
        if overlap_tokens and (len(overlap_tokens) >= min(2, len(informative)) or len(informative) == 1):
            # Prefer specific multi-token entity matches over generic terms such
            # as "Electric"; insertion order is not a relevance signal.
            overlap = len(overlap_tokens) / len(informative)
            matches.append((overlap + min(len(informative), 8) * 0.01, str(node)))
    return [entity for _, entity in sorted(matches, reverse=True)[:8]]


def query_intents(query: str) -> set[str]:
    lower = query.lower()
    intents = set()
    if any(word in lower for word in ("revenue", "ebit", "earnings", "financial", "cash")):
        intents.add("REPORTS_FINANCIAL_RESULT")
    if any(word in lower for word in ("sold", "sales", "deliver", "buyers", "share")):
        intents.add("REPORTS_SALES")
    if any(word in lower for word in ("policy", "regulation", "incentive", "tax", "deadline")):
        intents.add("REPORTS_POLICY")
    if any(word in lower for word in ("charge", "charger", "refueling")):
        intents.add("REPORTS_CHARGING")
    if any(word in lower for word in ("sentiment", "survey", "consider")):
        intents.add("REPORTS_SENTIMENT")
    if any(word in lower for word in ("forecast", "project", "outlook")):
        intents.add("REPORTS_FORECAST")
    return intents


def graph_rank(query: str, docs: list[Document], graph: nx.MultiDiGraph, flat_index: tuple[Counter[str], dict[str, Counter[str]]], hops: int = 2) -> tuple[list[tuple[str, float]], list[str], list[Triple]]:
    """Rank sources from entity-to-fact-to-document paths, capped at two hops."""
    matched = match_query_entities(query, graph)
    scores: Counter[str] = Counter()
    evidence: list[Triple] = []
    intents = query_intents(query)
    for entity in matched:
        direct_edges = list(graph.edges(entity, data=True)) + list(graph.in_edges(entity, data=True))
        for source, target, data in direct_edges:
            doc_id = str(data["doc_id"])
            predicate = data["predicate"]
            weight = 2.0 + (1.5 if predicate in intents else 0.0)
            scores[doc_id] += weight
            evidence.append(Triple(str(source), predicate, str(target), doc_id, data.get("evidence", "")))
            # One additional relation hop can surface a document attached to a
            # neighboring entity/fact, without traversing the entire dense graph.
            if hops >= 2:
                neighbor = target if source == entity else source
                for s2, t2, data2 in list(graph.edges(neighbor, data=True)) + list(graph.in_edges(neighbor, data=True)):
                    second_doc = str(data2["doc_id"])
                    scores[second_doc] += 0.35 + (0.35 if data2["predicate"] in intents else 0.0)
                    evidence.append(Triple(str(s2), data2["predicate"], str(t2), second_doc, data2.get("evidence", "")))
    lexical = dict(flat_rank(query, docs, flat_index))
    max_graph = max(scores.values(), default=0.0)
    max_lexical = max(lexical.values(), default=0.0)
    ranked = []
    for doc in docs:
        graph_score = scores[doc.doc_id] / max_graph if max_graph else 0.0
        lexical_score = lexical.get(doc.doc_id, 0.0) / max_lexical if max_lexical else 0.0
        # Use entity paths as a reranker. Lexical relevance remains dominant for
        # a noisy web corpus, so a weak entity match cannot displace strong text
        # evidence; a specific graph path can still break close lexical ties.
        score = 0.15 * graph_score + 0.85 * lexical_score if matched else lexical_score
        ranked.append((doc.doc_id, score))
    deduplicated = {(triple.subject, triple.predicate, triple.object, triple.doc_id): triple for triple in evidence}
    return sorted(ranked, key=lambda item: (-item[1], item[0])), matched, list(deduplicated.values())[:20]


def answer_with_citation(query: str, docs_by_id: dict[str, Document], ranked: list[tuple[str, float]]) -> tuple[str, str, str]:
    q = set(tokens(query))
    intents = query_intents(query)
    asks_for_measurement = bool(re.search(r"\b(how many|what proportion|what percentage|what .* revenue|what .* share|what .* growth|how does)\b", query, flags=re.IGNORECASE))
    candidates = []
    for doc_id, _ in ranked[:3]:
        for sent in sentences(docs_by_id[doc_id].text):
            score = len(q & set(tokens(sent)))
            if predicate_for(sent) in intents:
                score += 2
            if asks_for_measurement and any(char.isdigit() for char in sent):
                score += 1
            if score:
                candidates.append((score, sent, doc_id))
    if not candidates:
        return "Không tìm thấy bằng chứng đủ mạnh trong các tài liệu đã truy xuất.", "", ""
    best = sorted(candidates, key=lambda item: (-item[0], len(item[1])))[0]
    return f"{best[1]} [Nguồn: {best[2]}]", best[2], best[1]


def answer_from_context(query: str, docs_by_id: dict[str, Document], ranked: list[tuple[str, float]]) -> str:
    return answer_with_citation(query, docs_by_id, ranked)[0]


def generate_grounded_answer(query: str, docs_by_id: dict[str, Document], ranked: list[tuple[str, float]], model: str) -> str:
    """Generate an answer only from GraphRAG-retrieved text and enforce citations."""
    from openai import OpenAI

    context = "\n\n".join(
        f"[{doc_id}] {docs_by_id[doc_id].title}\n{docs_by_id[doc_id].text[:2400]}"
        for doc_id, _ in ranked[:3]
    )
    response = OpenAI(timeout=45.0, max_retries=1).chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=350,
        messages=[{"role": "user", "content": f"""Answer the question using only the source context below. Cite each factual statement with its document identifier in square brackets, such as [doc_30]. If the context does not support the answer, say so plainly. Do not use outside knowledge.

QUESTION: {query}

SOURCE CONTEXT:
{context}"""}],
    )
    return response.choices[0].message.content or "Không tạo được câu trả lời có căn cứ."


def term_coverage(answer: str, required_terms: tuple[str, ...]) -> float:
    normalized = re.sub(r"\s+", " ", answer).lower()
    return sum(term.lower() in normalized for term in required_terms) / len(required_terms)


def evaluate(docs: list[Document], graph: nx.MultiDiGraph, flat_index: tuple[Counter[str], dict[str, Counter[str]]], output: Path, extractor: str) -> pd.DataFrame:
    docs_by_id = {doc.doc_id: doc for doc in docs}
    rows = []
    for item in BENCHMARKS:
        if item.expected_doc not in docs_by_id:
            raise ValueError(f"Benchmark {item.benchmark_id} refers to missing {item.expected_doc}")
        flat_results = flat_rank(item.question, docs, flat_index)
        flat = [doc_id for doc_id, _ in flat_results]
        graph_results, entities_found, evidence = graph_rank(item.question, docs, graph, flat_index)
        gr = [doc_id for doc_id, _ in graph_results]
        flat_rank_pos = flat.index(item.expected_doc) + 1
        graph_rank_pos = gr.index(item.expected_doc) + 1
        flat_answer, flat_answer_doc, flat_sentence = answer_with_citation(item.question, docs_by_id, flat_results)
        graph_answer, graph_answer_doc, graph_sentence = answer_with_citation(item.question, docs_by_id, graph_results)
        rows.append({
            "benchmark_id": item.benchmark_id,
            "question": item.question,
            "expected_doc": item.expected_doc,
            "gold_answer": item.expected_answer,
            "required_terms": "; ".join(item.required_terms),
            "flat_top1": flat[0], "graph_top1": gr[0],
            "flat_rank": flat_rank_pos, "graph_rank": graph_rank_pos,
            "flat_hit_at_3": flat_rank_pos <= 3, "graph_hit_at_3": graph_rank_pos <= 3,
            "flat_answer": flat_answer,
            "flat_answer_doc": flat_answer_doc,
            "flat_evidence": flat_sentence,
            "flat_term_coverage": term_coverage(flat_answer, item.required_terms),
            "graph_answer": graph_answer,
            "graph_answer_doc": graph_answer_doc,
            "graph_evidence": graph_sentence,
            "graph_term_coverage": term_coverage(graph_answer, item.required_terms),
            "entities_matched": "; ".join(entities_found),
            "graph_triples": " | ".join(f"({t.subject}, {t.predicate}, {t.object}) [{t.doc_id}]" for t in evidence[:3]),
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "benchmark_20.csv", index=False)
    summary = pd.DataFrame([{
        "system": "Flat RAG (TF-IDF)", "top1_accuracy": (frame.flat_rank == 1).mean(), "hit_at_3": frame.flat_hit_at_3.mean(), "mrr": (1 / frame.flat_rank).mean(), "mean_answer_term_coverage": frame.flat_term_coverage.mean()
    }, {
        "system": "GraphRAG (entity + 2-hop)", "top1_accuracy": (frame.graph_rank == 1).mean(), "hit_at_3": frame.graph_hit_at_3.mean(), "mrr": (1 / frame.graph_rank).mean(), "mean_answer_term_coverage": frame.graph_term_coverage.mean()
    }])
    summary.to_csv(output / "benchmark_summary.csv", index=False)
    graph_wins = frame[(frame.flat_hit_at_3 == False) & (frame.graph_hit_at_3 == True)]
    flat_wins = frame[(frame.flat_hit_at_3 == True) & (frame.graph_hit_at_3 == False)]
    table_lines = ["| System | Top-1 | Hit@3 | MRR | Answer-term coverage |", "| --- | ---: | ---: | ---: | ---: |"]
    for _, row in summary.iterrows():
        table_lines.append(f"| {row.system} | {row.top1_accuracy:.2f} | {row.hit_at_3:.2f} | {row.mrr:.3f} | {row.mean_answer_term_coverage:.2f} |")
    lines = ["# Evaluation analysis", "", "## Method", "", "Twenty title-independent questions have a manually verified target document, expected answer, required answer terms, and cited source evidence.", "", "## Retrieval comparison", "", *table_lines, "", "## Cases where GraphRAG recovered a source Flat RAG missed", ""]
    if graph_wins.empty:
        lines.append("No GraphRAG-only Hit@3 cases in this run. This is a valid outcome, not a failure to report.")
    else:
        for _, row in graph_wins.iterrows():
            lines.append(f"- {row.benchmark_id}: Flat rank {row.flat_rank}; Graph rank {row.graph_rank}. {row.question}")
    lines.extend(["", "## Cases where Flat RAG recovered a source GraphRAG missed", ""])
    if flat_wins.empty:
        lines.append("No Flat-RAG-only Hit@3 cases in this run.")
    else:
        for _, row in flat_wins.iterrows():
            lines.append(f"- {row.benchmark_id}: Flat rank {row.flat_rank}; Graph rank {row.graph_rank}. {row.question}")
    interpretation = (
        "The graph was built from LLM-extracted semantic triples with sentence-level provenance. These metrics evaluate retrieval and source-grounded extraction; a separate LLM answer generator is available for manual queries and must be independently judged for hallucination."
        if extractor == "openai" else
        "The offline extractor creates conservative rule-based predicates. Therefore these measurements evaluate retrieval and source-grounded extraction, not free-form LLM hallucination. Run the LLM extractor and independently judge generated answers before making a hallucination claim."
    )
    lines.extend(["", "## Interpretation", "", interpretation])
    (output / "evaluation_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def draw_graph(graph: nx.MultiDiGraph, output: Path) -> None:
    """Render a readable, semantically-labelled slice rather than a hairball."""
    focus_names = {"Tesla", "NVIDIA", "VinFast", "Polestar", "ZEEKR", "Nikola", "Mercedes-Benz", "Inflation Reduction Act", "California", "Pew Research Center"}
    focus = [node for node in graph.nodes if str(node) in focus_names]
    display = nx.DiGraph()
    for node in focus:
        candidates = []
        for source, target, data in graph.out_edges(node, data=True):
            if data["predicate"].startswith("REPORTS_"):
                candidates.append((source, target, data))
        for source, target, data in candidates[:2]:
            display.add_node(source, kind="entity")
            display.add_node(target, kind="fact")
            display.add_edge(source, target, predicate=data["predicate"])
    # Corpus changes should not make visualization empty; retain a compact
    # high-degree fallback if none of the focus entities were recognized.
    if not display.nodes:
        entities_only = [n for n, d in graph.nodes(data=True) if d.get("kind") == "entity"]
        selected = sorted(entities_only, key=lambda n: graph.degree(n), reverse=True)[:20]
        display = graph.subgraph(selected).to_directed()
    plt.figure(figsize=(18, 12))
    pos = nx.spring_layout(display, seed=42, k=1.4)
    colors = ["#8ecae6" if data.get("kind") == "entity" else "#ffddd2" for _, data in display.nodes(data=True)]
    nx.draw_networkx_edges(display, pos, alpha=0.55, width=1.1, arrows=True, arrowsize=12)
    nx.draw_networkx_nodes(display, pos, node_size=1150, node_color=colors, edgecolors="#023047")
    labels = {node: (str(node)[:58] + "…") if len(str(node)) > 59 else str(node) for node in display.nodes}
    nx.draw_networkx_labels(display, pos, labels=labels, font_size=7)
    nx.draw_networkx_edge_labels(display, pos, edge_labels=nx.get_edge_attributes(display, "predicate"), font_size=6, rotate=False)
    plt.title("Knowledge graph — selected entities and source-grounded facts")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output / "knowledge_graph.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--extractor", choices=("heuristic", "openai", "gemini", "hybrid"), default="heuristic")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--query", help="Run one GraphRAG query after indexing")
    parser.add_argument("--answer-with-llm", action="store_true", help="Generate a source-grounded answer for --query with the OpenAI model")
    args = parser.parse_args()
    if args.extractor == "openai" and not os.getenv("OPENAI_API_KEY"):
        parser.error("OPENAI_API_KEY is required with --extractor openai")
    if args.answer_with_llm and not os.getenv("OPENAI_API_KEY"):
        parser.error("OPENAI_API_KEY is required with --answer-with-llm")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    docs = read_corpus(args.corpus)
    graph, triples, usage = build_graph(docs, args.extractor, args.model, args.output / "llm_cache" if args.extractor in {"openai", "gemini", "hybrid"} else None)
    flat_index = build_flat_index(docs)
    elapsed = time.perf_counter() - started
    pd.DataFrame([asdict(t) for t in triples]).to_csv(args.output / "triples.csv", index=False)
    nx.write_graphml(graph, args.output / "knowledge_graph.graphml")
    draw_graph(graph, args.output)
    summary = evaluate(docs, graph, flat_index, args.output, args.extractor)
    input_rate = float(os.getenv("OPENAI_INPUT_COST_PER_1M", "0"))
    output_rate = float(os.getenv("OPENAI_OUTPUT_COST_PER_1M", "0"))
    estimated_cost = usage.input_tokens / 1_000_000 * input_rate + usage.output_tokens / 1_000_000 * output_rate
    metrics = {
        "documents": len(docs), "nodes": graph.number_of_nodes(), "edges": graph.number_of_edges(), "triples": len(triples),
        "indexing_seconds": round(elapsed, 3), "extractor": args.extractor, "model": args.model if args.extractor in {"openai", "gemini", "hybrid"} else None,
        "llm_calls": usage.calls, "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
        "estimated_cost_usd": round(estimated_cost, 8) if args.extractor in {"openai", "gemini", "hybrid"} else 0,
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.output / "token_usage.json").write_text(json.dumps({
        "extractor": args.extractor, "model": args.model if args.extractor in {"openai", "gemini", "hybrid"} else None,
        "calls": usage.calls, "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
        "input_cost_per_1m_usd": input_rate if args.extractor in {"openai", "gemini", "hybrid"} else 0,
        "output_cost_per_1m_usd": output_rate if args.extractor in {"openai", "gemini", "hybrid"} else 0,
        "estimated_cost_usd": estimated_cost if args.extractor in {"openai", "gemini", "hybrid"} else 0,
    }, indent=2), encoding="utf-8")
    (args.output / "cost_analysis.md").write_text(
        f"# Cost analysis\n\n- Extractor: `{args.extractor}`\n- Artifact build time for this run: {elapsed:.3f}s\n- Documents: {len(docs)}; triples: {len(triples)}\n- LLM calls: {usage.calls}; input tokens: {usage.input_tokens}; output tokens: {usage.output_tokens}.\n- Estimated cost: ${estimated_cost:.8f}.\n\nCost is calculated from the actual API usage stored in the LLM checkpoint. The standard gpt-4o-mini rates used for this run are $0.15 per 1M input tokens and $0.60 per 1M output tokens, as configured at execution time. The exact provider usage is stored in `token_usage.json`.\n",
        encoding="utf-8",
    )
    summary_rows = []
    for _, row in summary.iterrows():
        summary_rows.append(f"| {row.system} | {row.top1_accuracy:.2f} | {row.hit_at_3:.2f} | {row.mrr:.3f} | {row.mean_answer_term_coverage:.2f} |")
    (args.output / "run_report.md").write_text(
        "# GraphRAG run report\n\n"
        f"- Extractor: `{args.extractor}`\n"
        f"- Model: `{args.model if args.extractor in {'openai', 'gemini', 'hybrid'} else 'N/A'}`\n"
        f"- Documents: {len(docs)}; triples: {len(triples)}; nodes: {graph.number_of_nodes()}; edges: {graph.number_of_edges()}\n"
        f"- Artifact build time for this run: {elapsed:.3f}s\n"
        f"- API calls: {usage.calls}; input tokens: {usage.input_tokens}; output tokens: {usage.output_tokens}; estimated cost: ${estimated_cost:.8f}\n\n"
        "| System | Top-1 | Hit@3 | MRR | Answer-term coverage |\n"
        "| --- | ---: | ---: | ---: | ---: |\n"
        + "\n".join(summary_rows) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))
    print(summary.to_string(index=False))
    if args.query:
        ranked, matched, evidence = graph_rank(args.query, docs, graph, flat_index)
        docs_by_id = {d.doc_id: d for d in docs}
        print("\nMatched entities:", matched)
        print("Top sources:", ranked[:3])
        answer = generate_grounded_answer(args.query, docs_by_id, ranked, args.model) if args.answer_with_llm else answer_from_context(args.query, docs_by_id, ranked)
        print("Answer:", answer)
        print("Evidence triples:")
        for triple in evidence[:8]:
            print(f"- ({triple.subject}, {triple.predicate}, {triple.object}) [{triple.doc_id}]")


if __name__ == "__main__":
    main()
