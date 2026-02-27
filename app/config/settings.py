"""Application settings and configuration"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Application
    APP_NAME: str = "DevPort Crawler"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Database
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/devportdb"

    # LLM API for summarization (OpenAI gpt-5-nano)
    OPENAI_API_KEY: Optional[str] = None
    LLM_MAX_TOKENS: int = 8000  # Max completion tokens (output) per request

    # GitHub API
    GITHUB_TOKEN: Optional[str] = None

    # Global project limits
    PORT_PROJECT_GLOBAL_TARGET: int = 1000

    PORT_CANDIDATE_WEIGHT_RELEVANCE: float = 0.5
    PORT_CANDIDATE_WEIGHT_STARS: float = 0.3
    PORT_CANDIDATE_WEIGHT_ACTIVITY: float = 0.2
    PORT_CANDIDATE_DIVERSITY_SOFT_CAP: int = 3
    PORT_CANDIDATE_MIN_STARS: int = 50000

    # Port-domain practical full-history caps
    PORT_BACKFILL_FULL_HISTORY: bool = True
    PORT_BACKFILL_MAX_STARGAZER_PAGES: int = 300
    PORT_BACKFILL_MAX_RELEASE_PAGES: int = 50
    PORT_BACKFILL_MAX_TAG_PAGES: int = 50
    PORT_BACKFILL_CHANGELOG_MAX_CHARS: int = 120000
    PORT_METRICS_HISTORY_DAYS_CAP: int = 730

    # Port-domain stage cadence controls
    PORT_PROJECT_SYNC_HOURS: int = 24
    PORT_EVENT_SYNC_HOURS: int = 24
    PORT_METRICS_SYNC_HOURS: int = 24
    PORT_STAR_HISTORY_SYNC_HOURS: int = 24

    # Port-domain GitHub client resilience controls
    PORT_GITHUB_TIMEOUT_SECONDS: float = 30.0
    PORT_GITHUB_MAX_RETRIES: int = 3
    PORT_GITHUB_BACKOFF_BASE_SECONDS: float = 1.0
    PORT_GITHUB_BACKOFF_MAX_SECONDS: float = 16.0
    PORT_GITHUB_RATE_LIMIT_BUFFER_SECONDS: int = 2
    PORT_GITHUB_CONCURRENCY: int = 4

    # Port-domain summarization retry policy
    PORT_SUMMARY_MAX_ATTEMPTS: int = 5
    PORT_SUMMARY_BACKOFF_BASE_SECONDS: float = 2.0
    PORT_SUMMARY_BACKOFF_MAX_SECONDS: float = 30.0
    PORT_SUMMARY_TIMEOUT_SECONDS: int = 45

    # Artificial Analysis API (for LLM rankings)
    ARTIFICIAL_ANALYSIS_API_KEY: Optional[str] = None

    # LLM Media Benchmarks (Artificial Analysis)
    # Uses the same API key; kept for clarity and future overrides
    ARTIFICIAL_ANALYSIS_MEDIA_API_KEY: Optional[str] = None

    # Crawling settings
    CRAWL_DELAY_SECONDS: int = 2
    MAX_CONCURRENT_REQUESTS: int = 5
    USER_AGENT: str = "DevPortCrawler/1.0 (+https://devport.kr)"

    # Playwright settings
    PLAYWRIGHT_HEADLESS: bool = True
    PLAYWRIGHT_TIMEOUT: int = 30000  # milliseconds

    # Deduplication
    TITLE_SIMILARITY_THRESHOLD: float = 0.9

    # Scoring
    GITHUB_SOURCE_WEIGHT: float = 2.0
    BLOG_SOURCE_WEIGHT: float = 1.0
    TIME_DECAY_DAYS: int = 7

    # Scoring - Time Decay (Exponential Decay System)
    SCORE_PLATEAU_DAYS: int = 2  # Days before score decay starts (fresh content plateau)
    SCORE_HALF_LIFE_DAYS: float = 4.0  # Exponential decay rate (days for score to halve)
    SCORE_MAX_AGE_DAYS: int = 14  # Articles older than this get zero score (hard cutoff)

    # Filtering
    MIN_REACTIONS_DEVTO: int = 10
    MIN_REACTIONS_HASHNODE: int = 5
    MIN_UPVOTES_REDDIT: int = 100
    MIN_SCORE_HACKERNEWS: int = 50
    MIN_STARS_GITHUB: int = 50

    # Hacker News settings
    MAX_STORIES_HACKERNEWS: int = 100  # Number of top stories to fetch
    MAX_AGE_DAYS_HACKERNEWS: int = 7  # Skip stories older than this

    # Reddit API (optional OAuth; falls back to public if missing)
    REDDIT_CLIENT_ID: Optional[str] = None
    REDDIT_CLIENT_SECRET: Optional[str] = None

    # Discord webhook for failed content fetch notifications
    DISCORD_WEBHOOK_URL: Optional[str] = None

    # API webhook handoff for crawler completion signals
    CRAWLER_WEBHOOK_URL: Optional[str] = None
    CRAWLER_WEBHOOK_SECRET: Optional[str] = None
    CRAWLER_WEBHOOK_TIMEOUT_SECONDS: float = 10.0
    CRAWLER_WEBHOOK_MAX_RETRIES: int = 3

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Ignore extra fields in .env


settings = Settings()
