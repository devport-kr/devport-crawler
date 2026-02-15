"""Project event model aligned to DevPort Port API contract."""

from datetime import datetime
import enum

from sqlalchemy import BigInteger, Boolean, Column, Date, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import relationship

from app.config.database import Base


class EventType(str, enum.Enum):
    """Event type values must stay compatible with DevPort API enum."""

    FEATURE = "feature"
    FIX = "fix"
    SECURITY = "security"
    BREAKING = "breaking"
    PERF = "perf"
    MISC = "misc"


class ProjectEvent(Base):
    """Release/event timeline entry mapped to `project_events` table."""

    __tablename__ = "project_events"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    external_id = Column(String(100), unique=True, nullable=False, index=True)
    project_id = Column(BigInteger, ForeignKey("projects.id"), nullable=False, index=True)

    version = Column(String(50), nullable=False)
    released_at = Column(Date, nullable=False)
    event_types = Column(ARRAY(String(20)), nullable=False, default=list)
    summary = Column(Text, nullable=False)
    bullets = Column(ARRAY(Text), nullable=True)
    impact_score = Column(Integer, nullable=True)
    is_security = Column(Boolean, nullable=False, default=False)
    is_breaking = Column(Boolean, nullable=False, default=False)
    source_url = Column(String(500), nullable=True)
    raw_notes = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    project = relationship("Project", backref="events")

    __table_args__ = (
        Index("idx_project_events_project_released", "project_id", "released_at"),
        Index("idx_project_events_security", "is_security"),
        Index("idx_project_events_breaking", "is_breaking"),
    )

    def __repr__(self):
        return f"<ProjectEvent {self.project_id}:{self.version}>"
