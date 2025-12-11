"""
AWS Lambda entrypoint for DevPort Crawler

Pure event-driven Lambda handler triggered by EventBridge Scheduler.
No FastAPI or HTTP server logic - just direct function invocation.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from app.orchestrator import CrawlerOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Instantiate orchestrator once per Lambda execution environment
orchestrator = CrawlerOrchestrator()


def lambda_handler(event: Optional[Dict[str, Any]], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda entrypoint for the DevPort crawler.

    Dispatches to specific crawler functions based on `event["source"]`.

    Expected event payloads (from EventBridge Scheduler or other callers):
    - {"source": "github"}
    - {"source": "hashnode"}
    - {"source": "medium"}
    - {"source": "reddit"}
    - {"source": "llm_rankings"}
    - {"source": "all_blogs"}

    Default is "all_blogs" if no source is provided.

    Args:
        event: Event payload from EventBridge or other AWS service
        context: Lambda context object

    Returns:
        Dictionary with statusCode, source, and result
    """
    # Extract source from event (default to "all_blogs")
    source = (event or {}).get("source", "all_blogs")
    logger.info(f"Lambda invoked with source: {source}")

    try:
        # Dispatch to appropriate crawler
        if source == "github":
            result = asyncio.run(orchestrator.run_github_crawler())

        elif source == "hashnode":
            result = asyncio.run(orchestrator.run_hashnode_crawler())

        elif source == "medium":
            result = asyncio.run(orchestrator.run_medium_crawler())

        elif source == "reddit":
            result = asyncio.run(orchestrator.run_reddit_crawler())

        elif source == "llm_rankings":
            result = asyncio.run(orchestrator.run_llm_crawler())

        elif source == "all_blogs":
            result = asyncio.run(orchestrator.run_all_crawlers())

        else:
            error_msg = f"Unknown source: {source}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"Crawler completed successfully: {result}")

        return {
            "statusCode": 200,
            "source": source,
            "result": result,
        }

    except Exception as e:
        logger.error(f"Lambda execution failed: {e}", exc_info=True)
        return {
            "statusCode": 500,
            "source": source,
            "error": str(e),
        }


# Allow local testing via `python -m app.handler`
if __name__ == "__main__":
    print("=" * 60)
    print("DevPort Crawler - Local Test")
    print("=" * 60)

    # Example test runs
    test_events = [
        {"source": "all_blogs"},
        {"source": "github"},
        {"source": "hashnode"},
        {"source": "medium"},
        {"source": "reddit"},
        {"source": "llm_rankings"},
    ]

    # Run a single test (change index to test different sources)
    test_event = {"source": "all_blogs"}
    print(f"\nTesting with event: {test_event}")
    print("-" * 60)

    result = lambda_handler(test_event, None)

    print("\nResult:")
    print(result)
    print("=" * 60)
