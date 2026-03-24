"""AWS Lambda entrypoint — pure event-driven, no HTTP server required."""

import asyncio
import logging
from typing import Any, Dict

from app.orchestrator import CrawlerOrchestrator
from app.jobs.port_sync import parse_project_ids, run_port_daily_sync

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Lazy-init: created on first invocation, reused across warm Lambda invocations.
# Avoids running constructors (env-var validation, API client init) during INIT phase.
_orchestrator = None


def _get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = CrawlerOrchestrator()
    return _orchestrator


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda handler for EventBridge-scheduled and manually-invoked crawls.

    Expected event format:
        {"source": "<source>"}               — run a specific crawler
        {"source": "refresh_scores", "days": 14}  — optional days param
        {"source": "port_sync", "stages": "events,metrics", "project_ids": "1,2"}

    Supported sources:
        github, devto, hashnode, reddit, hackernews,
        all_blogs, llm_rankings, llm_media_rankings,
        refresh_scores, port_sync
    """
    logger.info(f"Lambda invoked with event: {event}")

    source = event.get("source", "all_blogs")

    try:
        if source == "github":
            result = asyncio.run(_get_orchestrator().run_github_crawler())

        elif source == "devto":
            result = asyncio.run(_get_orchestrator().run_devto_crawler())

        elif source == "hashnode":
            result = asyncio.run(_get_orchestrator().run_hashnode_crawler())

        elif source == "reddit":
            result = asyncio.run(_get_orchestrator().run_reddit_crawler())

        elif source == "hackernews":
            result = asyncio.run(_get_orchestrator().run_hackernews_crawler())

        elif source == "all_blogs":
            result = asyncio.run(_get_orchestrator().run_all_crawlers())

        elif source == "llm_rankings":
            result = asyncio.run(_get_orchestrator().run_llm_crawler())

        elif source == "llm_media_rankings":
            result = asyncio.run(_get_orchestrator().run_llm_media_crawler())

        elif source == "refresh_scores":
            days = event.get("days")
            result = asyncio.run(_get_orchestrator().refresh_scores(days=days))

        elif source == "port_sync":
            result = asyncio.run(
                run_port_daily_sync(
                    stages=event.get("stages"),
                    project_ids=parse_project_ids(event.get("project_ids")),
                )
            )

        else:
            logger.warning(f"Unknown source: {source}")
            return {"statusCode": 400, "source": source, "error": f"Unknown source: {source}"}

        logger.info(f"Crawl completed for source={source}: {result}")
        return {"statusCode": 200, "source": source, "result": result}

    except Exception as e:
        logger.error(f"Lambda handler failed for source={source}: {e}", exc_info=True)
        return {"statusCode": 500, "source": source, "error": str(e)}


if __name__ == "__main__":
    # Local testing: python -m app.handler
    import sys

    source = sys.argv[1] if len(sys.argv) > 1 else "all_blogs"
    print(f"Running locally with source={source}")
    result = lambda_handler({"source": source}, None)
    print(result)
