"""
Debug script to test all crawlers and save raw results to local files.

No LLM calls, no database, no summarization — just pure fetch/crawl.
Results are saved as JSON files in the debug_output/ directory.

Usage:
    python debug_fetch.py              # Run all crawlers
    python debug_fetch.py devto        # Run specific crawler
    python debug_fetch.py devto hn     # Run multiple crawlers
Available crawler names:
    devto, hashnode, reddit, hn (hackernews), github
"""

import asyncio
import json
import os
import sys
import logging
from datetime import datetime
from typing import List, Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("debug_fetch")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "debug_output")


def raw_article_to_dict(article) -> Dict[str, Any]:
    """Convert a RawArticle to a JSON-serializable dict."""
    content = article.content or ""
    return {
        "title_en": article.title_en,
        "url": article.url,
        "source": article.source,
        "published_at": article.published_at.isoformat() if article.published_at else None,
        "external_id": article.external_id,
        "tags": article.tags,
        "content_length": len(content),
        "content_preview": content[:500] if content else "",
        "content_full": content,
        "stars": article.stars,
        "comments": article.comments,
        "upvotes": article.upvotes,
        "read_time": article.read_time,
        "language": article.language,
        "used_playwright": article.raw_data.get("used_playwright", False),
    }


def save_results(name: str, articles: list, elapsed: float):
    """Save crawler results to a JSON file."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data = {
        "crawler": name,
        "fetched_at": datetime.utcnow().isoformat(),
        "elapsed_seconds": round(elapsed, 2),
        "total_articles": len(articles),
        "articles_with_content": sum(1 for a in articles if a.get("content_full")),
        "avg_content_length": (
            round(sum(a["content_length"] for a in articles) / len(articles))
            if articles else 0
        ),
        "articles": articles,
    }

    filepath = os.path.join(OUTPUT_DIR, f"{name}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(articles)} articles to {filepath}")


def print_summary(name: str, articles: list, elapsed: float):
    """Print a short summary table for a crawler run."""
    from app.crawlers.base import MIN_CONTENT_LENGTH

    total = len(articles)
    with_content = sum(1 for a in articles if a.get("content_full"))
    avg_len = (
        round(sum(a["content_length"] for a in articles) / total)
        if total else 0
    )
    max_len = max((a["content_length"] for a in articles), default=0)
    min_len = min((a["content_length"] for a in articles), default=0)

    pw_retried = sum(1 for a in articles if a.get("used_playwright"))
    still_failed = sum(
        1 for a in articles
        if a["content_length"] < MIN_CONTENT_LENGTH and a.get("content_full") is not None
    )

    print(f"\n{'='*60}")
    print(f"  {name.upper()} — {elapsed:.1f}s")
    print(f"{'='*60}")
    print(f"  Total articles:      {total}")
    print(f"  With content:        {with_content} ({with_content*100//max(total,1)}%)")
    print(f"  Content length:      min={min_len}  avg={avg_len}  max={max_len}")
    if pw_retried or still_failed:
        print(f"  Playwright retried:  {pw_retried}")
        print(f"  Still failed (<{MIN_CONTENT_LENGTH}): {still_failed}")

    if articles:
        print(f"\n  Top 3 articles:")
        for a in articles[:3]:
            title = a["title_en"][:55]
            pw_tag = " [PW]" if a.get("used_playwright") else ""
            print(f"    - [{a['content_length']:>6} chars] {title}{pw_tag}")
    print()


# ── Crawler runners ──────────────────────────────────────────


async def run_devto() -> list:
    from app.crawlers.devto import DevToCrawler
    crawler = DevToCrawler()
    articles = await crawler.crawl()
    return [raw_article_to_dict(a) for a in articles]


async def run_hashnode() -> list:
    from app.crawlers.hashnode import HashnodeCrawler
    crawler = HashnodeCrawler()
    articles = await crawler.crawl()
    return [raw_article_to_dict(a) for a in articles]


# async def run_medium() -> list:
#     from app.crawlers.medium import MediumCrawler
#     crawler = MediumCrawler()
#     articles = await crawler.crawl()
#     return [raw_article_to_dict(a) for a in articles]


async def run_reddit() -> list:
    from app.crawlers.reddit import RedditCrawler
    crawler = RedditCrawler()
    articles = await crawler.crawl()
    return [raw_article_to_dict(a) for a in articles]


async def run_hackernews() -> list:
    from app.crawlers.hackernews import HackerNewsCrawler
    crawler = HackerNewsCrawler()
    articles = await crawler.crawl()
    return [raw_article_to_dict(a) for a in articles]


async def run_github() -> list:
    from app.crawlers.github import GitHubCrawler
    crawler = GitHubCrawler()
    articles = await crawler.crawl()
    return [raw_article_to_dict(a) for a in articles]


CRAWLERS = {
    "devto":     run_devto,
    "hashnode":  run_hashnode,
    # "medium":    run_medium,  # Disabled: ~70% of articles are RSS excerpts, not full content
    "reddit":    run_reddit,
    "hn":        run_hackernews,
    "github":    run_github,
}


async def run_crawler(name: str, func):
    """Run a single crawler, time it, save results."""
    logger.info(f"Starting {name}...")
    start = asyncio.get_event_loop().time()
    try:
        articles = await func()
    except Exception as e:
        logger.error(f"{name} FAILED: {e}", exc_info=True)
        return
    elapsed = asyncio.get_event_loop().time() - start

    save_results(name, articles, elapsed)
    print_summary(name, articles, elapsed)


async def main():
    args = [a.lower() for a in sys.argv[1:]]

    args = [a for a in args if not a.startswith("--")]

    # If specific crawlers requested, run only those
    if args:
        selected = {a: CRAWLERS[a] for a in args if a in CRAWLERS}
        unknown = [a for a in args if a not in CRAWLERS]
        if unknown:
            print(f"Unknown crawlers: {unknown}")
            print(f"Available: {list(CRAWLERS.keys())}")
            return
    else:
        selected = dict(CRAWLERS)

    print(f"\nRunning crawlers: {list(selected.keys())}")
    print(f"Output directory: {OUTPUT_DIR}\n")

    for name, func in selected.items():
        await run_crawler(name, func)

    # Final summary
    print(f"\n{'='*60}")
    print(f"  ALL DONE — results saved to {OUTPUT_DIR}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
