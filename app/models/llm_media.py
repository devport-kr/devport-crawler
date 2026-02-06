"""LLM media benchmark models"""

from datetime import datetime
from sqlalchemy import (
    Column,
    BigInteger,
    String,
    DateTime,
    DECIMAL,
    Integer,
    ForeignKey,
    Index,
)
from sqlalchemy.orm import relationship, declared_attr

from app.config.database import Base


class LLMMediaBase(Base):
    """Base fields for LLM media models"""

    __abstract__ = True

    id = Column(BigInteger, primary_key=True, index=True)

    # API identifiers
    external_id = Column(String(100), unique=True, nullable=False, index=True)  # API "id"
    slug = Column(String(200), unique=True, nullable=True, index=True)

    # Model identification
    name = Column(String(200), nullable=False)

    # Shared model creator (LLM provider)
    @declared_attr
    def model_creator_id(cls):  # type: ignore[override]
        return Column(BigInteger, ForeignKey("model_creators.id"), nullable=True, index=True)

    @declared_attr
    def model_creator(cls):  # type: ignore[override]
        return relationship("ModelCreator")

    # Rankings and stats
    elo = Column(DECIMAL(10, 2), nullable=True)
    model_rank = Column("model_rank", Integer, nullable=True, index=True)
    ci95 = Column(String(20), nullable=True)
    appearances = Column(Integer, nullable=True)

    # Keep release_date as string to preserve original format (e.g., "2025-04", "Apr 2025")
    release_date = Column(String(50), nullable=True)

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


class LLMMediaTextToImage(LLMMediaBase):
    __tablename__ = "llm_text_to_image_models"

    categories = relationship(
        "LLMMediaTextToImageCategory",
        back_populates="model",
        cascade="all, delete-orphan",
    )

    __table_args__ = (Index("idx_llm_tti_model_rank", "model_rank"),)

    def __repr__(self):
        return f"<LLMMediaTextToImage {self.external_id}: {self.name}>"


class LLMMediaImageEditing(LLMMediaBase):
    __tablename__ = "llm_image_editing_models"

    __table_args__ = (Index("idx_llm_ie_model_rank", "model_rank"),)

    def __repr__(self):
        return f"<LLMMediaImageEditing {self.external_id}: {self.name}>"


class LLMMediaTextToSpeech(LLMMediaBase):
    __tablename__ = "llm_text_to_speech_models"

    __table_args__ = (Index("idx_llm_tts_model_rank", "model_rank"),)

    def __repr__(self):
        return f"<LLMMediaTextToSpeech {self.external_id}: {self.name}>"


class LLMMediaTextToVideo(LLMMediaBase):
    __tablename__ = "llm_text_to_video_models"

    categories = relationship(
        "LLMMediaTextToVideoCategory",
        back_populates="model",
        cascade="all, delete-orphan",
    )

    __table_args__ = (Index("idx_llm_ttv_model_rank", "model_rank"),)

    def __repr__(self):
        return f"<LLMMediaTextToVideo {self.external_id}: {self.name}>"


class LLMMediaImageToVideo(LLMMediaBase):
    __tablename__ = "llm_image_to_video_models"

    categories = relationship(
        "LLMMediaImageToVideoCategory",
        back_populates="model",
        cascade="all, delete-orphan",
    )

    __table_args__ = (Index("idx_llm_itv_model_rank", "model_rank"),)

    def __repr__(self):
        return f"<LLMMediaImageToVideo {self.external_id}: {self.name}>"


class LLMMediaTextToImageCategory(Base):
    __tablename__ = "llm_text_to_image_categories"

    id = Column(BigInteger, primary_key=True, index=True)
    model_id = Column(
        BigInteger,
        ForeignKey("llm_text_to_image_models.id"),
        nullable=False,
        index=True,
    )

    style_category = Column(String(100), nullable=True)
    subject_matter_category = Column(String(100), nullable=True)

    elo = Column(DECIMAL(10, 2), nullable=True)
    ci95 = Column(String(20), nullable=True)
    appearances = Column(Integer, nullable=True)

    model = relationship("LLMMediaTextToImage", back_populates="categories")


class LLMMediaTextToVideoCategory(Base):
    __tablename__ = "llm_text_to_video_categories"

    id = Column(BigInteger, primary_key=True, index=True)
    model_id = Column(
        BigInteger,
        ForeignKey("llm_text_to_video_models.id"),
        nullable=False,
        index=True,
    )

    style_category = Column(String(100), nullable=True)
    subject_matter_category = Column(String(100), nullable=True)
    format_category = Column(String(100), nullable=True)

    elo = Column(DECIMAL(10, 2), nullable=True)
    ci95 = Column(String(20), nullable=True)
    appearances = Column(Integer, nullable=True)

    model = relationship("LLMMediaTextToVideo", back_populates="categories")


class LLMMediaImageToVideoCategory(Base):
    __tablename__ = "llm_image_to_video_categories"

    id = Column(BigInteger, primary_key=True, index=True)
    model_id = Column(
        BigInteger,
        ForeignKey("llm_image_to_video_models.id"),
        nullable=False,
        index=True,
    )

    style_category = Column(String(100), nullable=True)
    subject_matter_category = Column(String(100), nullable=True)
    format_category = Column(String(100), nullable=True)

    elo = Column(DECIMAL(10, 2), nullable=True)
    ci95 = Column(String(20), nullable=True)
    appearances = Column(Integer, nullable=True)

    model = relationship("LLMMediaImageToVideo", back_populates="categories")
