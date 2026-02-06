"""LLM Media Rankings crawler using Artificial Analysis API"""

from __future__ import annotations

from typing import Dict, Any, List, Tuple
from decimal import Decimal, InvalidOperation
from datetime import datetime
import logging
import re

import httpx
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.crawlers.base import BaseCrawler
from app.config.settings import settings
from app.models.model_creator import ModelCreator
from app.models.llm_media import (
    LLMMediaTextToImage,
    LLMMediaImageEditing,
    LLMMediaTextToSpeech,
    LLMMediaTextToVideo,
    LLMMediaImageToVideo,
    LLMMediaTextToImageCategory,
    LLMMediaTextToVideoCategory,
    LLMMediaImageToVideoCategory,
)

logger = logging.getLogger(__name__)


class LLMMediaRankingsCrawler(BaseCrawler):
    """
    Crawler for LLM media benchmark rankings using Artificial Analysis API

    Endpoints:
    - /data/media/text-to-image (include_categories=true)
    - /data/media/image-editing
    - /data/media/text-to-speech
    - /data/media/text-to-video (include_categories=true)
    - /data/media/image-to-video (include_categories=true)
    """

    API_BASE_URL = "https://artificialanalysis.ai/api/v2"

    MEDIA_TYPES = {
        "text_to_image": {
            "endpoint": "/data/media/text-to-image",
            "include_categories": True,
            "model": LLMMediaTextToImage,
            "category_model": LLMMediaTextToImageCategory,
        },
        "image_editing": {
            "endpoint": "/data/media/image-editing",
            "include_categories": False,
            "model": LLMMediaImageEditing,
            "category_model": None,
        },
        "text_to_speech": {
            "endpoint": "/data/media/text-to-speech",
            "include_categories": False,
            "model": LLMMediaTextToSpeech,
            "category_model": None,
        },
        "text_to_video": {
            "endpoint": "/data/media/text-to-video",
            "include_categories": True,
            "model": LLMMediaTextToVideo,
            "category_model": LLMMediaTextToVideoCategory,
        },
        "image_to_video": {
            "endpoint": "/data/media/image-to-video",
            "include_categories": True,
            "model": LLMMediaImageToVideo,
            "category_model": LLMMediaImageToVideoCategory,
        },
    }

    def __init__(self, db: Session | None = None):
        super().__init__()
        self.db = db

    async def crawl(self) -> Dict[str, Any]:
        """
        Fetch LLM media model data from Artificial Analysis API

        Returns:
            Dictionary with:
                - "creators": List[ModelCreator]
                - "media": Dict[str, List[LLMMedia*]]
        """
        self.log_start()

        creators_dict: Dict[str, ModelCreator] = {}
        media_results: Dict[str, List[Any]] = {}

        try:
            for media_type in self.MEDIA_TYPES:
                api_data = await self._fetch_api_data(media_type)
                models = []

                for model_data in api_data:
                    try:
                        creator = self._parse_model_creator(model_data.get("model_creator", {}))
                        if creator:
                            creator_key = creator.external_id or f"slug:{creator.slug}"
                            creators_dict[creator_key] = creator

                        model, categories = self._parse_media_model(media_type, model_data, creator)
                        model._media_categories = categories
                        models.append(model)
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to parse {media_type} model {model_data.get('id', 'unknown')}: {e}"
                        )
                        continue

                media_results[media_type] = models

            creators = list(creators_dict.values())
            total_models = sum(len(v) for v in media_results.values())
            self.logger.info(
                f"Successfully parsed {len(creators)} creators and {total_models} media models"
            )
        except Exception as e:
            self.log_error(e)
            return {"creators": [], "media": {}}

        self.log_end(total_models)
        return {"creators": creators, "media": media_results}

    async def _fetch_api_data(self, media_type: str) -> List[Dict[str, Any]]:
        """Fetch media models from Artificial Analysis API"""
        config = self.MEDIA_TYPES[media_type]
        endpoint = config["endpoint"]
        include_categories = config["include_categories"]

        url = f"{self.API_BASE_URL}{endpoint}"
        if include_categories:
            url = f"{url}?include_categories=true"

        api_key = settings.ARTIFICIAL_ANALYSIS_MEDIA_API_KEY or settings.ARTIFICIAL_ANALYSIS_API_KEY
        headers = {"x-api-key": api_key}

        self.logger.info(f"Fetching media data ({media_type}) from {url}")

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()

        if isinstance(data, dict) and "error" in data:
            self.logger.error(f"API returned error: {data.get('error')}")
            return []

        if isinstance(data, dict) and "data" in data:
            return data.get("data", [])

        return data.get("models", []) if isinstance(data, dict) else data

    def _parse_model_creator(self, api_creator: Dict[str, Any]) -> ModelCreator | None:
        if not api_creator:
            return None

        external_id = api_creator.get("id")
        name = api_creator.get("name") or api_creator.get("slug")
        slug = api_creator.get("slug")

        if not slug:
            base = name or "provider"
            slug = self._slugify(base)
            if external_id:
                slug = f"{slug}-{external_id[:8]}"

        if not name:
            name = slug

        if not (name and slug):
            self.logger.warning(f"Incomplete creator data: {api_creator}")
            return None

        creator = ModelCreator(
            external_id=external_id,
            slug=slug,
            name=name,
        )
        return creator

    def _parse_media_model(
        self, media_type: str, api_model: Dict[str, Any], creator: ModelCreator | None
    ) -> Tuple[Any, List[Any]]:
        model_cls = self.MEDIA_TYPES[media_type]["model"]

        external_id = api_model.get("id")
        slug = api_model.get("slug")
        name = api_model.get("name")

        if not external_id:
            raise ValueError("Missing model id")
        if not name:
            raise ValueError("Missing model name")

        elo = self._to_decimal(api_model.get("elo"))
        model_rank = self._to_int(api_model.get("rank"))
        ci95 = self._to_optional_string(api_model.get("ci95"))
        appearances = self._to_int(api_model.get("appearances"))
        release_date = api_model.get("release_date")

        model = model_cls(
            external_id=external_id,
            slug=slug,
            name=name,
            elo=elo,
            model_rank=model_rank,
            ci95=ci95,
            appearances=appearances,
            release_date=release_date,
        )

        if creator:
            if creator.external_id:
                model._creator_external_id = creator.external_id
            else:
                model._creator_slug = creator.slug

        categories = self._parse_categories(media_type, api_model.get("categories", []) or [])

        return model, categories

    def _parse_categories(self, media_type: str, api_categories: List[Dict[str, Any]]) -> List[Any]:
        category_cls = self.MEDIA_TYPES[media_type]["category_model"]
        if not category_cls:
            return []

        categories = []
        for item in api_categories:
            if media_type == "text_to_image":
                category = category_cls(
                    style_category=item.get("style_category"),
                    subject_matter_category=item.get("subject_matter_category"),
                    elo=self._to_decimal(item.get("elo")),
                    ci95=self._to_optional_string(item.get("ci95")),
                    appearances=item.get("appearances"),
                )
            else:
                category = category_cls(
                    style_category=item.get("style_category"),
                    subject_matter_category=item.get("subject_matter_category"),
                    format_category=item.get("format_category"),
                    elo=self._to_decimal(item.get("elo")),
                    ci95=self._to_optional_string(item.get("ci95")),
                    appearances=item.get("appearances"),
                )

            categories.append(category)

        return categories

    def _to_decimal(self, value: Any) -> Decimal | None:
        if value is None:
            return None
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned == "" or cleaned.lower() in {"-", "—", "–", "n/a", "na", "null"}:
                return None
            cleaned = cleaned.replace(",", "")
            value = cleaned
        try:
            return Decimal(str(value))
        except (ValueError, TypeError, InvalidOperation):
            return None

    def _to_int(self, value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned == "" or cleaned.lower() in {"-", "—", "–", "n/a", "na", "null"}:
                return None
            cleaned = cleaned.replace(",", "")
            try:
                return int(float(cleaned))
            except (ValueError, TypeError):
                return None
        try:
            return int(value)
        except (ValueError, TypeError):
            return None

    def _to_optional_string(self, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned == "" or cleaned.lower() in {"-", "—", "–", "n/a", "na", "null"}:
                return None
            return cleaned
        try:
            return str(value)
        except Exception:
            return None

    def _slugify(self, value: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
        return slug or "provider"

    def _ensure_unique_slug(self, model_cls: Any, slug: str | None, external_id: str) -> str | None:
        if not slug:
            return None
        existing = self.db.query(model_cls).filter(model_cls.slug == slug).first()
        if existing and existing.external_id != external_id:
            return f"{slug}-{external_id[:8]}"
        return slug

    def should_skip(self, article) -> bool:
        return False

    async def save_data(self, data: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
        """
        Save or update model creators and media models in database

        Returns:
            Dict with per-media counts saved
        """
        if not self.db:
            self.logger.warning("No database session provided, cannot save data")
            return {}

        creators = data.get("creators", [])
        media = data.get("media", {})

        creators_saved = 0
        creator_id_map_by_external: Dict[str, int] = {}
        creator_id_map_by_slug: Dict[str, int] = {}

        for creator in creators:
            try:
                existing = None
                if creator.external_id:
                    existing = self.db.query(ModelCreator).filter(
                        ModelCreator.external_id == creator.external_id
                    ).first()
                if not existing:
                    existing = self.db.query(ModelCreator).filter(
                        ModelCreator.slug == creator.slug
                    ).first()

                if existing:
                    existing.slug = creator.slug
                    existing.name = creator.name
                    existing.external_id = creator.external_id or existing.external_id
                    existing.updated_at = datetime.utcnow()
                    if existing.external_id:
                        creator_id_map_by_external[existing.external_id] = existing.id
                    creator_id_map_by_slug[existing.slug] = existing.id
                    self.logger.info(f"Updated creator: {existing.slug}")
                else:
                    self.db.add(creator)
                    self.db.flush()
                    if creator.external_id:
                        creator_id_map_by_external[creator.external_id] = creator.id
                    creator_id_map_by_slug[creator.slug] = creator.id
                    self.logger.info(f"Inserted new creator: {creator.slug}")

                creators_saved += 1
            except Exception as e:
                self.logger.error(f"Failed to save creator {creator.slug}: {e}")
                continue

        try:
            self.db.commit()
        except Exception as e:
            self.db.rollback()
            self.logger.error(f"Failed to commit creators to database: {e}")
            return {}

        results: Dict[str, Dict[str, int]] = {
            "creators": {"saved": creators_saved}
        }

        for media_type, models in media.items():
            model_cls = self.MEDIA_TYPES[media_type]["model"]
            categories_saved = 0
            models_saved = 0

            for model in models:
                try:
                    with self.db.begin_nested():
                        creator_external_id = getattr(model, "_creator_external_id", None)
                        creator_slug = getattr(model, "_creator_slug", None)
                        if creator_external_id and creator_external_id in creator_id_map_by_external:
                            model.model_creator_id = creator_id_map_by_external[creator_external_id]
                        elif creator_slug and creator_slug in creator_id_map_by_slug:
                            model.model_creator_id = creator_id_map_by_slug[creator_slug]

                        existing = self.db.query(model_cls).filter(
                            model_cls.external_id == model.external_id
                        ).first()

                        if existing:
                            existing.slug = self._ensure_unique_slug(
                                model_cls, model.slug, existing.external_id
                            )
                            existing.name = model.name
                            existing.model_creator_id = model.model_creator_id
                            existing.elo = model.elo
                            existing.model_rank = model.model_rank
                            existing.ci95 = model.ci95
                            existing.appearances = model.appearances
                            existing.release_date = model.release_date
                            existing.updated_at = datetime.utcnow()

                            # Replace categories on each update
                            if hasattr(existing, "categories"):
                                for category in list(existing.categories):
                                    self.db.delete(category)

                                for category in getattr(model, "_media_categories", []):
                                    existing.categories.append(category)
                                    categories_saved += 1

                            self.logger.info(f"Updated {media_type} model: {existing.external_id}")
                        else:
                            if hasattr(model, "_creator_external_id"):
                                delattr(model, "_creator_external_id")
                            if hasattr(model, "_creator_slug"):
                                delattr(model, "_creator_slug")

                            model.slug = self._ensure_unique_slug(
                                model_cls, model.slug, model.external_id
                            )
                            self.db.add(model)
                            self.db.flush()

                            for category in getattr(model, "_media_categories", []):
                                model.categories.append(category)
                                categories_saved += 1

                            self.logger.info(f"Inserted new {media_type} model: {model.external_id}")

                        models_saved += 1
                except IntegrityError as e:
                    self.db.rollback()
                    self.logger.error(
                        f"Integrity error saving {media_type} model {model.external_id}: {e}"
                    )
                    continue
                except Exception as e:
                    self.db.rollback()
                    self.logger.error(f"Failed to save {media_type} model {model.external_id}: {e}")
                    continue

            results[media_type] = {
                "saved": models_saved,
                "categories_saved": categories_saved,
            }

        try:
            self.db.commit()
        except Exception as e:
            self.db.rollback()
            self.logger.error(f"Failed to commit media models to database: {e}")

            for media_type in media.keys():
                if media_type in results:
                    results[media_type]["saved"] = 0
                    results[media_type]["categories_saved"] = 0

        return results
