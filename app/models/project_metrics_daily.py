"""Project daily metrics model aligned to DevPort Port API contract."""

from sqlalchemy import BigInteger, Column, Date, ForeignKey, Integer, UniqueConstraint
from sqlalchemy.orm import relationship

from app.config.database import Base


class ProjectMetricsDaily(Base):
    """Daily project metrics snapshots mapped to `project_metrics_daily` table."""

    __tablename__ = "project_metrics_daily"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    project_id = Column(BigInteger, ForeignKey("projects.id"), nullable=False)
    date = Column(Date, nullable=False)

    stars = Column(Integer, nullable=True)
    forks = Column(Integer, nullable=True)
    open_issues = Column(Integer, nullable=True)
    contributors = Column(Integer, nullable=True)

    project = relationship("Project", backref="metrics_daily")

    __table_args__ = (
        UniqueConstraint("project_id", "date", name="uk_project_metrics_daily"),
    )

    def __repr__(self):
        return f"<ProjectMetricsDaily {self.project_id}:{self.date}>"
