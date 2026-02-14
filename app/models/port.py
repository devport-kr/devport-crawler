"""Port model aligned to DevPort Port API contract."""

from datetime import datetime

from sqlalchemy import BigInteger, Column, DateTime, Index, Integer, String, Text

from app.config.database import Base


class Port(Base):
    """Port community entity mapped to `ports` table."""

    __tablename__ = "ports"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    external_id = Column(String(100), unique=True, nullable=False, index=True)
    port_number = Column(Integer, unique=True, nullable=False)
    slug = Column(String(50), unique=True, nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    accent_color = Column(String(7), nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("idx_ports_slug", "slug"),
    )

    def __repr__(self):
        return f"<Port {self.slug} ({self.port_number})>"
