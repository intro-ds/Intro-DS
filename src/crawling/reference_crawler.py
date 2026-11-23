"""Semantic Scholar reference crawler module.

Fetches structured citation and reference metadata via the Semantic Scholar API.
Requests are paced by a thread-safe rate limiter and issued from a small thread
pool, so network latency overlaps while the global request rate stays within the
API limits (1 req/s with an API key, 100 req/5 min without).
"""

from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Deque, Dict, List, Optional
import urllib.error
import urllib.parse
import urllib.request

from tqdm import tqdm

S2_BASE_URL = "https://api.semanticscholar.org/graph/v1/paper"
S2_FIELDS = "references.title,references.authors,references.publicationDate,references.externalIds,references.paperId"
MIN_REQUEST_INTERVAL = 1.1  # unauthenticated: just above 1 req/sec
AUTH_MIN_INTERVAL = 1.05  # API key: just above the guaranteed 1 req/sec
WINDOW_SECONDS = 300.0  # 5 minutes
MAX_REQUESTS_PER_WINDOW = 95  # safe margin below the unauthenticated limit of 100
DEFAULT_AUTH_WORKERS = 4  # threads used when an API key is available
MAX_RETRIES = 6
INITIAL_BACKOFF = 2.0
MAX_BACKOFF = 60.0
RETRYABLE_HTTP_CODES = {500, 502, 503, 504}
CHECKPOINT_EVERY = 25


class S2RateLimiter:
    """Thread-safe limiter that spaces the START of consecutive requests.

    Each caller reserves a time slot under a lock and sleeps outside the lock,
    so several threads can wait concurrently while the global rate is respected.
    An optional sliding window caps the number of requests per window.
    """

    def __init__(
        self,
        min_interval: float = MIN_REQUEST_INTERVAL,
        window_size: Optional[float] = None,
        max_in_window: Optional[int] = None,
    ):
        self.min_interval = min_interval
        self.window_size = window_size
        self.max_in_window = max_in_window
        self._lock = threading.Lock()
        self._next_allowed = 0.0
        self._timestamps: Deque[float] = deque()

    def wait(self) -> None:
        """Block until this caller's reserved slot is reached."""
        with self._lock:
            start = max(time.time(), self._next_allowed)
            if self.window_size and self.max_in_window:
                while self._timestamps and start - self._timestamps[0] > self.window_size:
                    self._timestamps.popleft()
                if len(self._timestamps) >= self.max_in_window:
                    start = max(start, self._timestamps[0] + self.window_size + 0.5)
                self._timestamps.append(start)
            self._next_allowed = start + self.min_interval

        delay = start - time.time()
        if delay > 0:
            time.sleep(delay)

    def penalize(self, seconds: float) -> None:
        """Pause ALL threads for `seconds` (used after 429 / 5xx responses)."""
        with self._lock:
            self._next_allowed = max(self._next_allowed, time.time() + seconds)


_unauth_limiter = S2RateLimiter(
    min_interval=MIN_REQUEST_INTERVAL,
    window_size=WINDOW_SECONDS,
    max_in_window=MAX_REQUESTS_PER_WINDOW,
)
_auth_limiter = S2RateLimiter(min_interval=AUTH_MIN_INTERVAL)


def _parse_reference_item(raw_ref: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a raw reference entry from Semantic Scholar API response.

    Args:
        raw_ref: Dict containing reference fields from S2.

    Returns:
        Structured reference dictionary.
    """
    ext_ids = raw_ref.get("externalIds") or {}
    ref_arxiv_id = ext_ids.get("ArXiv") or ext_ids.get("arxiv")

    authors_list: List[str] = []
    for auth in raw_ref.get("authors") or []:
        if isinstance(auth, dict) and auth.get("name"):
            authors_list.append(auth["name"])
        elif isinstance(auth, str):
            authors_list.append(auth)

    return {
        "ref_s2_id": raw_ref.get("paperId") or "",
        "ref_arxiv_id": ref_arxiv_id,
        "ref_title": raw_ref.get("title") or "",
        "ref_authors": authors_list,
        "ref_submission_date": raw_ref.get("publicationDate") or "",
        "has_arxiv_id": bool(ref_arxiv_id),
    }


def _retry_after_seconds(err: urllib.error.HTTPError) -> Optional[float]:
    """Read the Retry-After header of an HTTP error, if present and numeric."""
    try:
        value = err.headers.get("Retry-After") if err.headers else None
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def fetch_paper_references(
    arxiv_id: str,
    api_key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Query Semantic Scholar API for paper references with retries and rate limiting.

    Args:
        arxiv_id: Base arXiv ID, e.g. '2512.03231'.
        api_key: Optional Semantic Scholar API key.

    Returns:
        Structured reference record, or None on permanent failure.
    """
    key = api_key or os.environ.get("S2_API_KEY")
    query_params = urllib.parse.urlencode({"fields": S2_FIELDS})
    url = f"{S2_BASE_URL}/arXiv:{arxiv_id}?{query_params}"

    headers = {
        "User-Agent": "IntroDS-ReferenceCrawler/1.0 (Course Project Group 13)",
    }
    if key:
        headers["x-api-key"] = key

    req = urllib.request.Request(url, headers=headers)
    limiter = _auth_limiter if key else _unauth_limiter
    backoff = INITIAL_BACKOFF

    for _ in range(MAX_RETRIES):
        limiter.wait()

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                s2_paper_id = data.get("paperId", "")
                raw_refs = data.get("references") or []

                parsed_refs = [_parse_reference_item(r) for r in raw_refs]

                return {
                    "arxiv_id": arxiv_id,
                    "s2_paper_id": s2_paper_id,
                    "references": parsed_refs,
                }
        except urllib.error.HTTPError as err:
            if err.code == 404:
                # Paper not indexed yet by Semantic Scholar
                return {
                    "arxiv_id": arxiv_id,
                    "s2_paper_id": "",
                    "references": [],
                    "note": "not_found_on_s2",
                }
            if err.code == 429 or err.code in RETRYABLE_HTTP_CODES:
                # Rate limited or transient server error: pause every thread, then retry
                delay = min(_retry_after_seconds(err) or backoff, MAX_BACKOFF)
                limiter.penalize(delay)
                backoff = min(backoff * 2.0, MAX_BACKOFF)
                continue
            return None
        except Exception:
            # Network error / timeout / malformed JSON: back off and retry
            limiter.penalize(backoff)
            backoff = min(backoff * 2.0, MAX_BACKOFF)
            continue

    return None


def _write_json_atomic(path: Path, payload: Any) -> None:
    """Write JSON through a temp file so an interrupted run never leaves a partial file."""
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def crawl_references_for_paper(
    arxiv_id: str,
    output_dir: Path = Path("data/raw/corpus"),
    api_key: Optional[str] = None,
) -> bool:
    """Crawl and persist references for a given paper.

    Args:
        arxiv_id: Base arXiv ID.
        output_dir: Corpus root directory.
        api_key: Optional API key.

    Returns:
        True if fetched or cached successfully, False otherwise.
    """
    paper_dir = output_dir / arxiv_id
    refs_file = paper_dir / "refs.json"

    if refs_file.exists():
        return True

    paper_dir.mkdir(parents=True, exist_ok=True)
    res = fetch_paper_references(arxiv_id, api_key=api_key)
    if res is None:
        return False

    _write_json_atomic(refs_file, res)

    # Also save copy in each version directory if versions exist
    for v_dir in paper_dir.glob("v*"):
        if v_dir.is_dir():
            v_refs = v_dir / "refs.json"
            if not v_refs.exists():
                _write_json_atomic(v_refs, res)

    return True


def crawl_references_for_corpus(
    arxiv_ids: List[str],
    output_dir: Path = Path("data/raw/corpus"),
    api_key: Optional[str] = None,
    workers: Optional[int] = None,
) -> Dict[str, Any]:
    """Crawl references for all given paper IDs using a thread pool.

    Args:
        arxiv_ids: List of base arXiv IDs.
        output_dir: Corpus root directory.
        api_key: Optional API key (falls back to the S2_API_KEY environment variable).
        workers: Number of threads. Defaults to 4 with an API key, 1 without.

    Returns:
        Summary dict of crawled, cached, and failed counts.
    """
    key = api_key or os.environ.get("S2_API_KEY")
    if workers is None:
        workers = DEFAULT_AUTH_WORKERS if key else 1

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = output_dir / ".refs_checkpoint.json"

    completed: set[str] = set()
    pending: List[str] = []
    for aid in arxiv_ids:
        if (output_dir / aid / "refs.json").exists():
            completed.add(aid)
        else:
            pending.append(aid)

    cached_count = len(completed)
    crawled_count = 0
    failed_count = 0

    def _save_checkpoint() -> None:
        with checkpoint_file.open("w", encoding="utf-8") as f:
            json.dump(sorted(completed), f)

    pbar = tqdm(
        total=len(arxiv_ids),
        initial=cached_count,
        desc="Crawling References (S2 API)",
        unit="paper",
    )
    pbar.set_postfix({"cached": cached_count, "new": 0, "failed": 0})

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(crawl_references_for_paper, aid, output_dir, key): aid
            for aid in pending
        }
        try:
            for done_count, future in enumerate(as_completed(futures), start=1):
                aid = futures[future]
                try:
                    ok = future.result()
                except Exception:
                    ok = False

                if ok:
                    crawled_count += 1
                    completed.add(aid)
                else:
                    failed_count += 1

                pbar.update(1)
                pbar.set_postfix(
                    {"cached": cached_count, "new": crawled_count, "failed": failed_count}
                )
                if done_count % CHECKPOINT_EVERY == 0:
                    _save_checkpoint()
        except KeyboardInterrupt:
            # Stop queued work; finished papers are already saved and will be skipped next run
            executor.shutdown(wait=False, cancel_futures=True)
            _save_checkpoint()
            raise

    pbar.close()
    _save_checkpoint()

    return {
        "total": len(arxiv_ids),
        "cached": cached_count,
        "crawled": crawled_count,
        "failed": failed_count,
    }