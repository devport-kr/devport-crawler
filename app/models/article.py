"""Article model matching DevPort API schema"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import ARRAY
from datetime import datetime
import enum

from app.config.database import Base


class ItemType(str, enum.Enum):
    """Article item type"""
    REPO = "REPO"
    BLOG = "BLOG"
    DISCUSSION = "DISCUSSION"


class Source(str, enum.Enum):
    """Article source"""
    GITHUB = "github"
    DEVTO = "devto"
    MEDIUM = "medium"
    HACKERNEWS = "hackernews"


class Category(str, enum.Enum):
    """Article category"""
    AI_LLM = "AI_LLM"
    DEVOPS_SRE = "DEVOPS_SRE"
    INFRA_CLOUD = "INFRA_CLOUD"
    DATABASE = "DATABASE"
    BLOCKCHAIN = "BLOCKCHAIN"
    SECURITY = "SECURITY"
    DATA_SCIENCE = "DATA_SCIENCE"
    ARCHITECTURE = "ARCHITECTURE"
    MOBILE = "MOBILE"
    FRONTEND = "FRONTEND"
    BACKEND = "BACKEND"
    OTHER = "OTHER"


class Article(Base):
    """
    Article model matching DevPort API schema

    This model must match the Article entity in devport-api
    """
    __tablename__ = "articles"

    id = Column(Integer, primary_key=True, index=True)

    # External ID (unique identifier from source or generated UUID)
    external_id = Column(String(100), unique=True, nullable=False, index=True)

    # Core fields
    item_type = Column(String(20), nullable=False)  # VARCHAR in DB, not enum
    source = Column(String(20), nullable=False, index=True)  # VARCHAR in DB, not enum
    category = Column(String(20), nullable=False, index=True)  # VARCHAR in DB, not enum

    # Content fields
    summary_ko_title = Column(String(500), nullable=False)
    summary_ko_body = Column(Text, nullable=True)
    title_en = Column(String(500), nullable=False)
    url = Column(String(1000), nullable=False, unique=True)

    # Engagement
    score = Column(Integer, nullable=False, index=True)

    # Tags - NOTE: Stored in separate article_tags table in Spring Boot schema
    # tags = Column(ARRAY(String), nullable=True)  # Disabled for schema compatibility

    # Metadata (embedded fields)
    stars = Column(Integer, nullable=True)
    comments = Column(Integer, nullable=True)
    upvotes = Column(Integer, nullable=True)
    read_time = Column(String(50), nullable=True)
    language = Column(String(50), nullable=True)

    # Timestamps
    created_at_source = Column(DateTime, nullable=False, index=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<Article {self.id}: {self.title_en[:50]}>"
