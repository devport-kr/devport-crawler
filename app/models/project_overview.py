"""Project overview model aligned to DevPort Port API contract."""

from datetime import datetime

from sqlalchemy import BigInteger, Column, DateTime, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import relationship

from app.config.database import Base


class ProjectOverview(Base):
    """Project overview/document summary mapped to `project_overviews` table."""

    __tablename__ = "project_overviews"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    project_id = Column(BigInteger, ForeignKey("projects.id"), nullable=False, unique=True)

    summary = Column(Text, nullable=False)
    highlights = Column(ARRAY(Text), nullable=True, default=list)
    quickstart = Column(Text, nullable=True)
    links = Column(JSONB, nullable=True)
    source_url = Column(String(500), nullable=True)
    raw_hash = Column(String(64), nullable=True)
    fetched_at = Column(DateTime, nullable=True)
    summarized_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    project = relationship("Project", backref="overview", uselist=False)

    def __repr__(self):
        return f"<ProjectOverview {self.project_id}>"
