"""FastAPI application entry point"""

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
import asyncio
import logging
from datetime import datetime

from app.config.settings import settings
from app.orchestrator import CrawlerOrchestrator
from app.jobs.port_sync import parse_project_ids, run_port_backfill, run_port_daily_sync
from app.crawlers.base import RawArticle
from app.services.summarizer import SummarizerService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="DevPort Crawler",
    description="Web scraping service for developer content aggregation",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global orchestrator instance
orchestrator = CrawlerOrchestrator()
summarizer = SummarizerService()

# Store last run stats (in-memory, for simple deployment)
last_stats = {}


class SummarizeRequest(BaseModel):
    url: str
    title: str
    content: str


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "DevPort Crawler",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/api/health",
            "stats": "/api/stats",
            "crawl_devto": "POST /api/crawl/devto",
            "crawl_hashnode": "POST /api/crawl/hashnode",
            "crawl_medium": "POST /api/crawl/medium",
            "crawl_reddit": "POST /api/crawl/reddit",
            "crawl_hackernews": "POST /api/crawl/hackernews",
            "crawl_github": "POST /api/crawl/github",
            "crawl_llm": "POST /api/crawl/llm-rankings",
            "crawl_llm_media": "POST /api/crawl/llm-media",
            "crawl_port_sync": "POST /api/crawl/port-sync",
            "crawl_port_backfill": "POST /api/crawl/port-backfill",
            "crawl_all": "POST /api/crawl/all",
            "deduplicate": "POST /api/deduplicate",
            "refresh_scores": "POST /api/refresh-scores (HTTP) or {\"source\": \"refresh_scores\"} (Lambda)",
            "summarize": "POST /api/summarize"
        }
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint for serverless platforms"""
    return {
        "status": "healthy",
        "service": "devport-crawler",
        "version": "1.0.0"
    }


@app.get("/api/stats")
async def get_stats():
    """Get crawling statistics"""
    return last_stats.get("all", {
        "last_run": None,
        "articles_scraped": 0,
        "errors": 0,
        "sources": {}
    })


@app.post("/api/crawl/devto")
async def crawl_devto(background_tasks: BackgroundTasks):
    """Trigger Dev.to crawling"""
    logger.info("Dev.to crawl triggered")

    async def run_crawler():
        try:
            stats = await orchestrator.run_devto_crawler()
            last_stats["devto"] = stats
            logger.info(f"Dev.to crawler completed: {stats.get('saved', 0)} articles saved")
        except Exception as e:
            logger.error(f"Dev.to crawler failed: {e}", exc_info=True)

    background_tasks.add_task(run_crawler)
    return {
        "status": "started",
        "source": "devto",
        "message": "Dev.to crawler started in background"
    }


@app.post("/api/crawl/hashnode")
async def crawl_hashnode(background_tasks: BackgroundTasks):
    """Trigger Hashnode crawling"""
    logger.info("Hashnode crawl triggered")

    async def run_crawler():
        try:
            stats = await orchestrator.run_hashnode_crawler()
            last_stats["hashnode"] = stats
            logger.info(f"Hashnode crawler completed: {stats.get('saved', 0)} articles saved")
        except Exception as e:
            logger.error(f"Hashnode crawler failed: {e}", exc_info=True)

    background_tasks.add_task(run_crawler)
    return {
        "status": "started",
        "source": "hashnode",
        "message": "Hashnode crawler started in background"
    }


@app.post("/api/crawl/medium")
async def crawl_medium(background_tasks: BackgroundTasks):
    """Trigger Medium crawling"""
    logger.info("Medium crawl triggered")

    async def run_crawler():
        try:
            stats = await orchestrator.run_medium_crawler()
            last_stats["medium"] = stats
            logger.info(f"Medium crawler completed: {stats.get('saved', 0)} articles saved")
        except Exception as e:
            logger.error(f"Medium crawler failed: {e}", exc_info=True)

    background_tasks.add_task(run_crawler)
    return {
        "status": "started",
        "source": "medium",
        "message": "Medium crawler started in background"
    }


@app.post("/api/crawl/reddit")
async def crawl_reddit(background_tasks: BackgroundTasks):
    """Trigger Reddit crawling"""
    logger.info("Reddit crawl triggered")

    async def run_crawler():
        try:
            stats = await orchestrator.run_reddit_crawler()
            last_stats["reddit"] = stats
            logger.info(f"Reddit crawler completed: {stats.get('saved', 0)} articles saved")
        except Exception as e:
            logger.error(f"Reddit crawler failed: {e}", exc_info=True)

    background_tasks.add_task(run_crawler)
    return {
        "status": "started",
        "source": "reddit",
        "message": "Reddit crawler started in background"
    }


@app.post("/api/crawl/hackernews")
async def crawl_hackernews(background_tasks: BackgroundTasks):
    """Trigger Hacker News crawling"""
    logger.info("Hacker News crawl triggered")

    async def run_crawler():
        try:
            stats = await orchestrator.run_hackernews_crawler()
            last_stats["hackernews"] = stats
            logger.info(f"Hacker News crawler completed: {stats.get('saved', 0)} articles saved")
        except Exception as e:
            logger.error(f"Hacker News crawler failed: {e}", exc_info=True)

    background_tasks.add_task(run_crawler)
    return {
        "status": "started",
        "source": "hackernews",
        "message": "Hacker News crawler started in background"
    }


@app.post("/api/crawl/github")
async def crawl_github(background_tasks: BackgroundTasks):
    """Trigger GitHub trending crawling"""
    logger.info("GitHub crawl triggered")

    async def run_crawler():
        try:
            stats = await orchestrator.run_github_crawler()
            last_stats["github"] = stats
            logger.info(f"GitHub crawler completed: {stats.get('saved', 0)} articles saved")
        except Exception as e:
            logger.error(f"GitHub crawler failed: {e}", exc_info=True)

    background_tasks.add_task(run_crawler)
    return {
        "status": "started",
        "source": "github",
        "message": "GitHub crawler started in background"
    }


@app.post("/api/crawl/llm-rankings")
async def crawl_llm_rankings(background_tasks: BackgroundTasks):
    """Trigger LLM rankings crawling"""
    logger.info("LLM rankings crawl triggered")

    async def run_crawler():
        try:
            stats = await orchestrator.run_llm_crawler()
            last_stats["llm_rankings"] = stats
            logger.info(f"LLM crawler completed")
        except Exception as e:
            logger.error(f"LLM crawler failed: {e}", exc_info=True)

    background_tasks.add_task(run_crawler)
    return {
        "status": "started",
        "source": "llm_rankings",
        "message": "LLM rankings crawler started in background"
    }


@app.post("/api/crawl/llm-media")
async def crawl_llm_media(background_tasks: BackgroundTasks):
    """Trigger LLM media rankings crawling"""
    logger.info("LLM media rankings crawl triggered")

    async def run_crawler():
        try:
            stats = await orchestrator.run_llm_media_crawler()
            last_stats["llm_media_rankings"] = stats
            logger.info("LLM media crawler completed")
        except Exception as e:
            logger.error(f"LLM media crawler failed: {e}", exc_info=True)

    background_tasks.add_task(run_crawler)
    return {
        "status": "started",
        "source": "llm_media_rankings",
        "message": "LLM media rankings crawler started in background"
    }


@app.post("/api/crawl/port-sync")
async def crawl_port_sync(background_tasks: BackgroundTasks, stages: str | None = None, project_ids: str | None = None):
    """Trigger port-domain daily sync (events/metrics by default)."""
    parsed_project_ids = parse_project_ids(project_ids)
    logger.info("Port daily sync triggered", extra={"stages": stages, "project_ids": parsed_project_ids})

    async def run_sync() -> None:
        try:
            stats = await run_port_daily_sync(stages=stages, project_ids=parsed_project_ids)
            last_stats["port_sync"] = stats
            logger.info("Port daily sync completed")
        except Exception as e:
            logger.error(f"Port daily sync failed: {e}", exc_info=True)

    background_tasks.add_task(run_sync)
    return {
        "status": "started",
        "source": "port_sync",
        "stages": stages or "events,metrics",
        "project_ids": parsed_project_ids,
        "message": "Port daily sync started in background",
    }


@app.post("/api/crawl/port-backfill")
async def crawl_port_backfill(
    background_tasks: BackgroundTasks,
    stages: str | None = None,
    project_ids: str | None = None,
    requested_metrics_days: int = 3650,
):
    """Trigger resumable port-domain backfill with caps/checkpoint reporting."""
    parsed_project_ids = parse_project_ids(project_ids)
    logger.info(
        "Port backfill triggered",
        extra={"stages": stages, "project_ids": parsed_project_ids, "requested_metrics_days": requested_metrics_days},
    )

    async def run_backfill() -> None:
        try:
            stats = await run_port_backfill(
                stages=stages,
                project_ids=parsed_project_ids,
                requested_metrics_days=requested_metrics_days,
            )
            last_stats["port_backfill"] = stats
            logger.info("Port backfill completed")
        except Exception as e:
            logger.error(f"Port backfill failed: {e}", exc_info=True)

    background_tasks.add_task(run_backfill)
    return {
        "status": "started",
        "source": "port_backfill",
        "stages": stages or "projects,events,star_history,metrics",
        "project_ids": parsed_project_ids,
        "requested_metrics_days": requested_metrics_days,
        "message": "Port backfill started in background",
    }


@app.post("/api/crawl/all")
async def crawl_all(background_tasks: BackgroundTasks):
    """Trigger all blog crawlers"""
    logger.info("All crawlers triggered")

    async def run_all():
        try:
            stats = await orchestrator.run_all_crawlers()
            last_stats["all"] = stats
            logger.info(f"All crawlers completed: {stats.get('total_saved', 0)} total articles saved")
        except Exception as e:
            logger.error(f"All crawlers failed: {e}", exc_info=True)

    background_tasks.add_task(run_all)
    return {
        "status": "started",
        "sources": ["devto", "hashnode", "medium", "reddit"],
        "message": "All blog crawlers started in background"
    }


@app.post("/api/deduplicate")
async def deduplicate(background_tasks: BackgroundTasks):
    """Run deduplication on existing articles"""
    logger.info("Deduplication triggered")

    async def run_dedup():
        try:
            stats = await orchestrator.run_deduplication()
            last_stats["deduplication"] = stats
            logger.info(f"Deduplication completed: {stats.get('removed', 0)} duplicates removed")
        except Exception as e:
            logger.error(f"Deduplication failed: {e}", exc_info=True)

    background_tasks.add_task(run_dedup)
    return {
        "status": "started",
        "message": "Deduplication started in background"
    }


@app.post("/api/refresh-scores")
async def refresh_scores(background_tasks: BackgroundTasks, days: int = None):
    """
    Recalculate scores for articles within the scoring window

    This updates scores to reflect time decay as articles age.
    Should be run daily via cron/scheduler.

    Query params:
        days: Number of days back to refresh (default: uses SCORE_MAX_AGE_DAYS setting, typically 14)
    """
    from app.config.settings import settings
    effective_days = days if days is not None else getattr(settings, 'SCORE_MAX_AGE_DAYS', 14)
    logger.info(f"Score refresh triggered for last {effective_days} days")

    async def run_refresh():
        try:
            stats = await orchestrator.refresh_scores(days=days)
            last_stats["refresh_scores"] = stats
            logger.info(f"Score refresh completed: {stats.get('updated', 0)} articles updated")
        except Exception as e:
            logger.error(f"Score refresh failed: {e}", exc_info=True)

    background_tasks.add_task(run_refresh)
    return {
        "status": "started",
        "message": f"Score refresh started in background for articles from last {days} days"
    }


@app.post("/api/summarize")
async def summarize_article(request: SummarizeRequest):
    """
    Summarize a single article given its URL and full content.

    Returns Korean title, summary, category, and tags via LLM.
    """
    logger.info(f"Summarize request for: {request.url}")

    raw_article = RawArticle(
        title_en=request.title,
        url=request.url,
        source="manual",
        published_at=datetime.utcnow(),
        content=request.content,
    )

    try:
        results = await summarizer.summarize_batch([raw_article], batch_size=1, delay=0, max_tokens_override=128000)
    except Exception as e:
        logger.error(f"Summarization failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")

    result = results[0] if results else None
    if result is None:
        raise HTTPException(status_code=500, detail="LLM summarization returned no result")

    return {
        "url": request.url,
        "is_technical": result.get("is_technical", False),
        "title_ko": result.get("title_ko", ""),
        "summary_ko": result.get("summary_ko", ""),
        "category": result.get("category", "OTHER"),
        "tags": result.get("tags", []),
    }


# AWS Lambda handler
def lambda_handler(event: Dict[str, Any], context: Any):
    """
    AWS Lambda handler for scheduled events

    Expected event format:
    {
        "source": "github" | "all_blogs" | "llm_rankings" | "refresh_scores" | "port_sync" | "port_backfill",
        "days": 30  # Optional: for refresh_scores only
    }
    """
    logger.info(f"Lambda invoked with event: {event}")

    # If this is a scheduled event, trigger appropriate crawler
    if "source" in event:
        source = event["source"]
        logger.info(f"Scheduled crawl for source: {source}")

        # Run crawler synchronously in Lambda
        try:
            if source == "github":
                stats = asyncio.run(orchestrator.run_github_crawler())
                logger.info(f"GitHub crawl completed: {stats}")
                return {
                    "statusCode": 200,
                    "body": f"Crawled GitHub successfully: {stats.get('saved', 0)} saved"
                }

            elif source == "all_blogs":
                stats = asyncio.run(orchestrator.run_all_crawlers())
                logger.info(f"All blogs crawl completed: {stats}")
                return {
                    "statusCode": 200,
                    "body": f"Crawled all blogs successfully: {stats.get('total_saved', 0)} saved"
                }

            elif source == "llm_rankings":
                stats = asyncio.run(orchestrator.run_llm_crawler())
                logger.info(f"LLM rankings crawl completed: {stats}")
                return {
                    "statusCode": 200,
                    "body": f"Crawled LLM rankings successfully"
                }

            elif source == "refresh_scores":
                # Recalculate scores for articles from last N days (default: 30)
                days = event.get("days", 30)
                stats = asyncio.run(orchestrator.refresh_scores(days=days))
                logger.info(f"Score refresh completed: {stats}")
                return {
                    "statusCode": 200,
                    "body": f"Refreshed {stats.get('updated', 0)} article scores"
                }

            elif source == "port_sync":
                stats = asyncio.run(
                    run_port_daily_sync(
                        stages=event.get("stages"),
                        project_ids=parse_project_ids(event.get("project_ids")),
                    )
                )
                logger.info(f"Port sync completed: {stats}")
                return {
                    "statusCode": 200,
                    "body": stats,
                }

            elif source == "port_backfill":
                stats = asyncio.run(
                    run_port_backfill(
                        stages=event.get("stages"),
                        project_ids=parse_project_ids(event.get("project_ids")),
                        checkpoints=event.get("checkpoints"),
                        requested_metrics_days=int(event.get("requested_metrics_days", 3650)),
                    )
                )
                logger.info(f"Port backfill completed: {stats}")
                return {
                    "statusCode": 200,
                    "body": stats,
                }

            else:
                return {
                    "statusCode": 400,
                    "body": f"Unknown source: {source}"
                }

        except Exception as e:
            logger.error(f"Lambda crawler failed: {e}", exc_info=True)
            return {
                "statusCode": 500,
                "body": f"Crawler failed: {str(e)}"
            }

    # Otherwise, handle as HTTP request
    from mangum import Mangum
    handler = Mangum(app)
    return handler(event, context)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
