"""LLM Rankings crawler using Artificial Analysis API"""

from typing import List, Dict, Any, Tuple
import httpx
from app.crawlers.base import BaseCrawler
from app.config.settings import settings
from app.models.llm_model import LLMModel
from app.models.model_creator import ModelCreator
from sqlalchemy.orm import Session
from decimal import Decimal
from datetime import datetime, date
import logging

logger = logging.getLogger(__name__)


class LLMRankingsCrawler(BaseCrawler):
    """
    Crawler for LLM benchmark rankings using Artificial Analysis API

    Data Source: https://artificialanalysis.ai/
    API Docs: https://api.artificialanalysis.ai/docs
    """

    API_BASE_URL = "https://artificialanalysis.ai/api/v2"
    API_ENDPOINT = "/data/llms/models"

    def __init__(self, db: Session = None):
        super().__init__()
        self.db = db

    async def crawl(self) -> Dict[str, Any]:
        """
        Fetch LLM model data from Artificial Analysis API

        Returns:
            Dictionary with:
                - "creators": List of ModelCreator objects
                - "models": List of LLMModel objects
        """
        self.log_start()
        creators_dict = {}  #llm 제공사(OpenAI, Anthropic, ...) external_id로 저장
        models = []

        try:
            # api fetch
            api_data = await self._fetch_api_data()

            if not api_data or "models" not in api_data:
                self.logger.warning("No models data received from API")
                return {"creators": [], "models": []}

            # 반환받은 api data 파싱 및 변환
            for model_data in api_data["models"]:
                try:
                    # llm(model) 제공사 부터 파싱
                    creator = self._parse_model_creator(model_data.get("model_creator", {}))
                    if creator and creator.external_id:
                        creators_dict[creator.external_id] = creator

                    # llm(model) 파싱
                    model = self._parse_model(model_data, creator)
                    models.append(model)
                except Exception as e:
                    self.logger.warning(f"Failed to parse model {model_data.get('id', 'unknown')}: {e}")
                    continue

            creators = list(creators_dict.values())
            self.logger.info(f"Successfully parsed {len(creators)} creators and {len(models)} models")

        except Exception as e:
            self.log_error(e)
            return {"creators": [], "models": []}

        self.log_end(len(models))
        return {"creators": creators, "models": models}

    async def _fetch_api_data(self) -> Dict[str, Any]:
        """
        Fetch data from Artificial Analysis API

        Returns:
            JSON response data
        """
        url = f"{self.API_BASE_URL}{self.API_ENDPOINT}"

        headers = {
            "x-api-key": settings.ARTIFICIAL_ANALYSIS_API_KEY #https://artificialanalysis.ai에서 발급받은 api key
        }

        self.logger.info(f"Fetching LLM data from {url}")

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()

        # 에러 여부 확인
        if isinstance(data, dict) and "error" in data:
            self.logger.error(f"API returned error: {data.get('error')}")
            return {}

        # llm 모델 데이터 추출 (artificialanalysis.ai v2 api)
        if isinstance(data, dict) and "data" in data:
            models_data = data.get("data", [])
            self.logger.info(f"Received {len(models_data)} models from API")
            return {"models": models_data} 

        # API 구조 변경시 Fallback 
        self.logger.info(f"Received {len(data.get('models', []))} models from API")
        return data

    def _parse_model_creator(self, api_creator: Dict[str, Any]) -> ModelCreator | None:
        """
        Parse API response model_creator data into ModelCreator object

        Args:
            api_creator: Model creator data from API response
                Example: {"id": "d874d370-...", "name": "Alibaba", "slug": "alibaba"}

        Returns:
            ModelCreator object or None
        """
        if not api_creator:
            return None

        external_id = api_creator.get("id")
        name = api_creator.get("name")
        slug = api_creator.get("slug")

        if not (external_id and name and slug):
            self.logger.warning(f"Incomplete creator data: {api_creator}")
            return None

        creator = ModelCreator(
            external_id=external_id,
            slug=slug,
            name=name,
        )

        return creator

    def _parse_model(self, api_model: Dict[str, Any], creator: ModelCreator = None) -> LLMModel:
        """
        Parse API response model data into LLMModel object

        Args:
            api_model: Model data from API response
            creator: ModelCreator object (optional)

        Returns:
            LLMModel object
        """
        external_id = api_model.get("id")  # API UUID
        slug = api_model.get("slug")  # API slug
        model_id = slug or external_id  # slug를 model_id로 사용, fallback to external_id

        model_name = api_model.get("name")
        provider = api_model.get("model_creator", {}).get("name", "Unknown")  # Legacy field
        description = api_model.get("description")

        release_date_str = api_model.get("release_date")
        release_date = None
        if release_date_str:
            try:
                release_date = datetime.strptime(release_date_str, "%Y-%m-%d").date()
            except ValueError:
                self.logger.warning(f"Invalid release_date format: {release_date_str}")

        self.logger.debug(
            f"Parsing model: id={model_id}, slug={slug}, release_date={release_date}, "
            f"creator={creator.slug if creator else 'None'}"
        )

        pricing = api_model.get("pricing", {})
        price_input = self._to_decimal(pricing.get("price_1m_input_tokens"))
        price_output = self._to_decimal(pricing.get("price_1m_output_tokens"))
        price_blended = self._to_decimal(pricing.get("price_1m_blended_3_to_1"))

        context_window = api_model.get("context_window")
        license_type = api_model.get("license")
        output_speed_median = self._to_decimal(api_model.get("median_output_tokens_per_second"))
        latency_ttft = self._to_decimal(api_model.get("median_time_to_first_token_seconds"))
        median_time_to_first_answer_token = self._to_decimal(api_model.get("median_time_to_first_answer_token"))

        # 각 llm 모델 별 벤치마크 점수 추출
        # 모든 점수는 0.0~ 1.0에서 0-100 단위로 변환, 
        evaluations = api_model.get("evaluations", {})

        # Agentic
        score_terminal_bench_hard = self._convert_to_percentage(evaluations.get("terminalbench_hard"))
        score_tau_bench_telecom = self._convert_to_percentage(evaluations.get("tau2"))

        # Reasoning & Knowledge
        score_aa_lcr = self._convert_to_percentage(evaluations.get("lcr"))
        score_humanitys_last_exam = self._convert_to_percentage(evaluations.get("hle"))
        score_mmlu_pro = self._convert_to_percentage(evaluations.get("mmlu_pro")) # 해당 밴치마크 v2에서 제공 x
        score_gpqa_diamond = self._convert_to_percentage(evaluations.get("gpqa"))

        # Coding
        score_livecode_bench = self._convert_to_percentage(evaluations.get("livecodebench"))
        score_scicode = self._convert_to_percentage(evaluations.get("scicode"))

        # Specialized
        score_ifbench = self._convert_to_percentage(evaluations.get("ifbench"))
        score_math_500 = self._convert_to_percentage(evaluations.get("math_500"))
        score_aime = self._convert_to_percentage(evaluations.get("aime"))
        score_aime_2025 = self._convert_to_percentage(evaluations.get("aime_25"))  # API uses "aime_25"

        # Composite (artificialanalysis.ai에서 자체적으로 만든 밴치마크)
        score_aa_intelligence_index = self._to_decimal(
            evaluations.get("artificial_analysis_intelligence_index")
        )
        score_aa_coding_index = self._to_decimal(
            evaluations.get("artificial_analysis_coding_index")
        )
        score_aa_math_index = self._to_decimal(
            evaluations.get("artificial_analysis_math_index")
        )

        # LLMModel object 생성
        model = LLMModel(
            
            external_id=external_id,
            slug=slug,
            model_id=model_id,
            model_name=model_name,
            release_date=release_date,

            provider=provider,

            description=description,

            price_input=price_input,
            price_output=price_output,
            price_blended=price_blended,

            context_window=context_window,
            output_speed_median=output_speed_median,
            latency_ttft=latency_ttft,
            median_time_to_first_answer_token=median_time_to_first_answer_token,
            license=license_type,
            
            # Agentic
            score_terminal_bench_hard=score_terminal_bench_hard,
            score_tau_bench_telecom=score_tau_bench_telecom,
            # Reasoning & Knowledge
            score_aa_lcr=score_aa_lcr,
            score_humanitys_last_exam=score_humanitys_last_exam,
            score_mmlu_pro=score_mmlu_pro,
            score_gpqa_diamond=score_gpqa_diamond,
            # Coding
            score_livecode_bench=score_livecode_bench,
            score_scicode=score_scicode,
            # Specialized
            score_ifbench=score_ifbench,
            score_math_500=score_math_500,
            score_aime=score_aime,
            score_aime_2025=score_aime_2025,
            # Composite 
            score_aa_intelligence_index=score_aa_intelligence_index,
            score_aa_coding_index=score_aa_coding_index,
            score_aa_math_index=score_aa_math_index,
        )

        # 제공사 external_id llm_model <-> llm_provider 연동위해
        if creator:
            model._creator_external_id = creator.external_id

        return model

    def _to_decimal(self, value: Any) -> Decimal | None:
        """
        Safely convert value to Decimal

        Args:
            value: Value to convert

        Returns:
            Decimal value or None
        """
        if value is None:
            return None

        try:
            return Decimal(str(value))
        except (ValueError, TypeError):
            return None

    def _convert_to_percentage(self, value: Any) -> Decimal | None:
        """
        Convert API score to percentage (0-100 scale)

        API returns scores in two formats:
        - 0-1 decimals (e.g., 0.774 for 77.4%) - need to multiply by 100
        - 0-100 values (e.g., 36 for intelligence_index) - keep as is

        Args:
            value: Score value from API

        Returns:
            Decimal value in 0-100 range or None
        """
        if value is None:
            return None

        try:
            decimal_value = Decimal(str(value))

            if decimal_value <= 1:
                return (decimal_value * 100).quantize(Decimal('0.01'))

            return decimal_value.quantize(Decimal('0.01'))

        except (ValueError, TypeError):
            return None

    def should_skip(self, article) -> bool:
        """Not used for LLM rankings"""
        return False

    async def save_data(self, data: Dict[str, Any]) -> Dict[str, int]:
        """
        Save or update model creators and models in database

        Args:
            data: Dictionary with "creators" and "models" lists

        Returns:
            Dictionary with counts of creators and models saved
        """
        if not self.db:
            self.logger.warning("No database session provided, cannot save data")
            return {"creators": 0, "models": 0}

        creators = data.get("creators", [])
        models = data.get("models", [])

        creators_saved = 0
        models_saved = 0

        # Step 1: 제공사부터 저장 (with upsert)
        creator_id_map = {}  # Map external_id : database id

        for creator in creators:
            try:
                existing = self.db.query(ModelCreator).filter(
                    ModelCreator.external_id == creator.external_id
                ).first()

                if existing:
                    existing.slug = creator.slug
                    existing.name = creator.name
                    existing.updated_at = datetime.utcnow()
                    creator_id_map[creator.external_id] = existing.id
                    self.logger.info(f"Updated creator: {creator.slug}")
                else:
                    self.db.add(creator)
                    self.db.flush()  # Flush to get ID
                    creator_id_map[creator.external_id] = creator.id
                    self.logger.info(f"Inserted new creator: {creator.slug}")

                creators_saved += 1

            except Exception as e:
                self.logger.error(f"Failed to save creator {creator.slug}: {e}")
                continue

        # llm_provider 커밋
        try:
            self.db.commit()
        except Exception as e:
            self.db.rollback()
            self.logger.error(f"Failed to commit creators to database: {e}")
            return {"creators": 0, "models": 0}

        # Step 2: llm_models 저장 (with upsert and link to creators)
        for model in models:
            try:
                creator_external_id = getattr(model, '_creator_external_id', None)
                if creator_external_id and creator_external_id in creator_id_map:
                    model.model_creator_id = creator_id_map[creator_external_id]
                    self.logger.debug(
                        f"Linked model {model.model_id} to creator_id={model.model_creator_id}"
                    )

                # 저장전 로그
                self.logger.debug(
                    f"Model data: id={model.model_id}, slug={model.slug}, "
                    f"release_date={model.release_date}, model_creator_id={model.model_creator_id}"
                )

                existing = self.db.query(LLMModel).filter(
                    LLMModel.model_id == model.model_id
                ).first()

                if existing:
                    existing.slug = model.slug
                    existing.external_id = model.external_id
                    existing.release_date = model.release_date
                    existing.model_creator_id = model.model_creator_id
                    existing.model_name = model.model_name
                    existing.provider = model.provider
                    existing.description = model.description

                    existing.price_input = model.price_input
                    existing.price_output = model.price_output
                    existing.price_blended = model.price_blended

                    existing.context_window = model.context_window
                    existing.output_speed_median = model.output_speed_median
                    existing.latency_ttft = model.latency_ttft
                    existing.median_time_to_first_answer_token = model.median_time_to_first_answer_token
                    existing.license = model.license

                    existing.score_terminal_bench_hard = model.score_terminal_bench_hard
                    existing.score_tau_bench_telecom = model.score_tau_bench_telecom
                    existing.score_aa_lcr = model.score_aa_lcr
                    existing.score_humanitys_last_exam = model.score_humanitys_last_exam
                    existing.score_mmlu_pro = model.score_mmlu_pro
                    existing.score_gpqa_diamond = model.score_gpqa_diamond
                    existing.score_livecode_bench = model.score_livecode_bench
                    existing.score_scicode = model.score_scicode
                    existing.score_ifbench = model.score_ifbench
                    existing.score_math_500 = model.score_math_500
                    existing.score_aime = model.score_aime
                    existing.score_aime_2025 = model.score_aime_2025
                    existing.score_aa_intelligence_index = model.score_aa_intelligence_index
                    existing.score_aa_coding_index = model.score_aa_coding_index
                    existing.score_aa_math_index = model.score_aa_math_index

                    existing.updated_at = datetime.utcnow()

                    self.logger.info(
                        f"Updated model: {model.model_id} (slug={existing.slug}, "
                        f"release_date={existing.release_date}, creator_id={existing.model_creator_id})"
                    )
                else:
                    if hasattr(model, '_creator_external_id'):
                        delattr(model, '_creator_external_id')
                    self.db.add(model)
                    self.logger.info(
                        f"Inserted new model: {model.model_id} (slug={model.slug}, "
                        f"release_date={model.release_date}, creator_id={model.model_creator_id})"
                    )

                models_saved += 1

            except Exception as e:
                self.logger.error(f"Failed to save model {model.model_id}: {e}")
                continue

        # llm_model 커밋
        try:
            self.db.commit()
            self.logger.info(f"Successfully saved {creators_saved} creators and {models_saved} models")
        except Exception as e:
            self.db.rollback()
            self.logger.error(f"Failed to commit models to database: {e}")
            models_saved = 0

        return {"creators": creators_saved, "models": models_saved}
