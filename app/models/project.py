"""Project model aligned to DevPort Port API contract."""

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Column,
    Date,
    DateTime,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB

from app.config.database import Base


class Project(Base):
    """Project entity mapped to `projects` table."""

    __tablename__ = "projects"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    external_id = Column(String(100), unique=True, nullable=False, index=True)

    name = Column(String(100), nullable=False)
    full_name = Column(String(200), nullable=False)
    repo_url = Column(String(500), nullable=False)
    homepage_url = Column(String(500), nullable=True)
    description = Column(Text, nullable=True)

    stars = Column(Integer, nullable=False, default=0)
    stars_week_delta = Column(Integer, nullable=False, default=0)
    forks = Column(Integer, nullable=False, default=0)
    contributors = Column(Integer, nullable=False, default=0)

    language = Column(String(50), nullable=True)
    language_color = Column(String(7), nullable=True)
    license = Column(String(50), nullable=True)
    last_release = Column(Date, nullable=True)
    releases_30d = Column(Integer, nullable=False, default=0)
    tags = Column(JSONB, nullable=True)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("idx_projects_stars", "stars"),
    )

    def __repr__(self):
        return f"<Project {self.full_name}>"
