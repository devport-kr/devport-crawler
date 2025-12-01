"""FastAPI application entry point"""

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
import asyncio
import logging

from app.config.settings import settings
from app.orchestrator import CrawlerOrchestrator

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

# Store last run stats (in-memory, for simple deployment)
last_stats = {}


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
            "crawl_github": "POST /api/crawl/github",
            "crawl_llm": "POST /api/crawl/llm-rankings",
            "crawl_all": "POST /api/crawl/all",
            "deduplicate": "POST /api/deduplicate",
            "refresh_scores": "POST /api/refresh-scores?days=30"
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
        from app.crawlers.devto import DevToCrawler
        from app.crawlers.base import RawArticle

        try:
            crawler = DevToCrawler()
            articles = await crawler.crawl()
            saved = await orchestrator._process_and_save_articles(articles)
            last_stats["devto"] = {
                "crawled": len(articles),
                "saved": saved,
                "source": "devto"
            }
            logger.info(f"Dev.to crawler completed: {saved} articles saved")
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
        from app.crawlers.hashnode import HashnodeCrawler

        try:
            crawler = HashnodeCrawler()
            articles = await crawler.crawl()
            saved = await orchestrator._process_and_save_articles(articles)
            last_stats["hashnode"] = {
                "crawled": len(articles),
                "saved": saved,
                "source": "hashnode"
            }
            logger.info(f"Hashnode crawler completed: {saved} articles saved")
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
        from app.crawlers.medium import MediumCrawler

        try:
            crawler = MediumCrawler()
            articles = await crawler.crawl()
            saved = await orchestrator._process_and_save_articles(articles)
            last_stats["medium"] = {
                "crawled": len(articles),
                "saved": saved,
                "source": "medium"
            }
            logger.info(f"Medium crawler completed: {saved} articles saved")
        except Exception as e:
            logger.error(f"Medium crawler failed: {e}", exc_info=True)

    background_tasks.add_task(run_crawler)
    return {
        "status": "started",
        "source": "medium",
        "message": "Medium crawler started in background"
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
        "sources": ["devto", "hashnode", "medium"],
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
async def refresh_scores(background_tasks: BackgroundTasks, days: int = 30):
    """
    Recalculate scores for articles from last N days

    This updates scores to reflect time decay as articles age.
    Should be run daily via cron/scheduler.

    Query params:
        days: Number of days back to refresh (default: 30)
    """
    logger.info(f"Score refresh triggered for last {days} days")

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


# AWS Lambda handler
def lambda_handler(event: Dict[str, Any], context: Any):
    """
    AWS Lambda handler for scheduled events

    Expected event format:
    {
        "source": "github" | "all_blogs" | "llm_rankings"
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
