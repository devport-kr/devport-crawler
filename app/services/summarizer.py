"""Korean summarization service using LLM APIs"""

from typing import Dict, Optional
import asyncio
import json
import logging
import re
from openai import AsyncOpenAI
from app.config.settings import settings
from app.crawlers.base import RawArticle

logger = logging.getLogger(__name__)


class LLMQuotaExceeded(Exception):
    """Raised when the LLM provider reports quota exhaustion."""


class SummarizerService:
    """Service to generate Korean summaries using LLM APIs"""

    SYSTEM_MESSAGE = (
        "You are an expert English-to-Korean technical translator and editor "
        "specializing in software engineering content.\n\n"
        "## Your Task\n"
        "You produce **comprehensive Korean translations** of English tech articles. "
        "This is a TRANSLATION that is slightly condensed — NOT a summary or abstract. "
        "The Korean output should be approximately 70-80% of the original article's length. "
        "A Korean reader should fully understand the article without needing to read the original.\n\n"
        "## Korean Writing Rules\n"
        "- Write natural, fluent Korean as if the author originally wrote in Korean. "
        "Avoid literal translation patterns (e.g., '~하는 것이다', '~되어진다', '~할 수 있습니다' repetition).\n"
        "- Use industry-standard Korean terms (e.g., 'deployment'→'배포', 'scalability'→'확장성').\n"
        "- Keep proper nouns in English (React, Kubernetes, AWS, PostgreSQL, Kafka, etc.).\n"
        "- Preserve the original author's tone and voice: formal→formal, casual→casual, opinionated→opinionated.\n"
        "- Do NOT translate code blocks — include them exactly as-is.\n"
        "- Use markdown formatting (##, ###, -, **, `) to mirror the original structure.\n\n"
        "## Critical: Length and Detail\n"
        "- LONGER output is ALWAYS better than missing content. Never cut for brevity.\n"
        "- Include EVERY key argument, technical detail, example, and insight from the original.\n"
        "- Preserve the article's section structure, heading hierarchy, and logical flow.\n"
        "- Only trim genuinely redundant phrasing — never skip entire paragraphs or sections."
    )

    def __init__(self):
        self.max_tokens = self._resolve_max_tokens()

        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required")
        self.openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    def _resolve_max_tokens(self) -> int:
        requested = getattr(settings, "LLM_MAX_TOKENS", 8000) or 8000
        if not isinstance(requested, int) or requested <= 0:
            logger.warning(f"Invalid LLM_MAX_TOKENS={requested}; defaulting to 8000")
            requested = 8000

        hard_cap = 128000  # gpt-5-nano max completion tokens
        if requested > hard_cap:
            logger.warning(
                f"LLM_MAX_TOKENS={requested} exceeds gpt-5-nano limit {hard_cap}; clamping"
            )
        return min(requested, hard_cap)

    def _build_batch_prompt(self, articles: list[RawArticle]) -> str:
        """Build prompt for multiple articles at once"""
        articles_text = ""
        for i, article in enumerate(articles, 1):
            content = (article.content or "").strip()
            tags_str = ', '.join(article.tags[:10]) if article.tags else '(none)'

            articles_text += (
                f"--- Article {i} ---\n"
                f"Title: {article.title_en}\n"
                f"URL: {article.url}\n"
                f"Tags: {tags_str}\n"
            )
            if content:
                articles_text += f"Content:\n{content}\n"
            else:
                articles_text += "Content: (not available)\n"
            articles_text += "\n"

        prompt = f"""Translate the following {len(articles)} English developer article(s) into comprehensive Korean.

## Articles

{articles_text}

## Instructions

For each article, produce a JSON object with these 6 fields:

### 1. is_technical (boolean)
Is this article useful or interesting for software developers?
- TRUE: tutorials, code, architecture, dev tools, frameworks, system design, security, infra, AI/ML, developer career, tech startup products
- FALSE: pure politics, non-tech business, consumer product reviews, social issues unrelated to tech
- Ask yourself: "Would a developer building software care about this?"

### 2. title_ko (string, max 100 characters)
A concise Korean title capturing the core topic.

### 3. summary_ko (string, markdown format)
Write a **comprehensive Korean translation** of the full article. This is a faithful translation that is slightly condensed — NOT a summary, NOT an abstract, NOT a brief overview.

**Length target: aim for 70-80% of the original article's length.** Longer output is always better than missing content.

Requirements:
- Translate EVERY section of the article — do NOT skip or merge sections
- Preserve the original section structure, heading hierarchy (##, ###), and logical flow
- Include ALL key arguments, explanations, technical details, examples, and insights
- Include code examples from the original exactly as-is (do not translate code)
- Follow the author's original flow and order — do NOT reorder or restructure
- Preserve the author's tone and voice (if opinionated, keep the opinion; if humorous, keep the humor)
- A Korean reader must be able to fully understand the article without reading the English original
- Only trim genuinely redundant or repetitive phrasing — NEVER skip entire paragraphs or ideas
- If content is unavailable (title only), write a brief description based on the title. Do NOT fabricate details.

### 4. category (string)
Pick the single best match: AI_LLM, DEVOPS_SRE, INFRA_CLOUD, DATABASE, BLOCKCHAIN, SECURITY, DATA_SCIENCE, ARCHITECTURE, MOBILE, FRONTEND, BACKEND, OTHER

### 5. tags (array of strings)
3-5 lowercase tags. Use hyphens instead of spaces.

### 6. url (string)
Return the input URL exactly as given (used for matching).

## Output Format

Return a JSON array inside a ```json code fence. Maintain the same article order.
No text outside the code fence.

```json
[
  {{
    "url": "https://example.com/article1",
    "is_technical": true,
    "title_ko": "Python에서 비동기 처리 완벽 가이드",
    "summary_ko": "## 개요\\n\\n이 글은 Python의 asyncio 라이브러리를 활용한 비동기 처리 방법을 깊이 있게 다룹니다. 동시성(concurrency)과 병렬성(parallelism)의 차이를 명확히 구분하고, 실제 프로덕션 환경에서 async/await 패턴을 효과적으로 사용하는 방법을 설명합니다.\\n\\n## async/await 패턴의 기본 사용법\\n\\nPython 3.5에서 도입된 `async/await` 구문은 비동기 코드를 동기 코드처럼 읽기 쉽게 작성할 수 있게 해줍니다. 기본적인 패턴은 다음과 같습니다:\\n\\n```python\\nasync def fetch_data(url):\\n    async with aiohttp.ClientSession() as session:\\n        async with session.get(url) as response:\\n            return await response.json()\\n\\nasync def main():\\n    results = await asyncio.gather(\\n        fetch_data('https://api.example.com/users'),\\n        fetch_data('https://api.example.com/posts')\\n    )\\n```\\n\\n`asyncio.gather()`를 사용하면 여러 코루틴을 동시에 실행하여 I/O 바운드 작업에서 상당한 성능 향상을 얻을 수 있습니다. 저자는 실제 프로젝트에서 API 호출 시간을 60% 이상 단축한 사례를 공유합니다.\\n\\n## 동시성 vs 병렬성\\n\\n동시성은 여러 작업을 번갈아 처리하는 것이고, 병렬성은 여러 작업을 실제로 동시에 처리하는 것입니다. asyncio는 동시성을 제공하며, 이는 네트워크 요청이나 파일 I/O처럼 대기 시간이 긴 작업에 특히 효과적입니다.\\n\\n## 실전 팁과 주의사항\\n\\n저자는 CPU 바운드 작업에서는 asyncio 대신 `multiprocessing`을 사용할 것을 권장하며, 혼합 워크로드에서는 `loop.run_in_executor()`를 활용한 하이브리드 접근법을 제안합니다. 또한 에러 처리, 타임아웃 설정, 디버깅 기법 등 프로덕션 환경에서 겪는 현실적인 문제와 해결책을 상세히 다룹니다.",
    "category": "BACKEND",
    "tags": ["python", "async", "concurrency"]
  }}
]
```

JSON rules:
- Newlines inside summary_ko must be \\n (escaped)
- Quotes inside strings must be \\" (escaped)
- Return ONLY valid JSON — no trailing commas, no comments"""

        return prompt

    def _parse_batch_response(self, content: str, articles: list[RawArticle]) -> list[Dict[str, str]]:
        """Parse LLM batch response into list of structured data"""
        try:
            data_array = self._safe_json_loads(content, expect_array=True)
            if data_array is None:
                logger.error("Failed to parse LLM batch response after all repair attempts")
                logger.error(f"Problematic content (first 1000 chars): {content[:1000]}")
                logger.error(f"Problematic content (last 500 chars): {content[-500:]}")
                return [None] * len(articles)

            # Handle structured output wrapper: {"articles": [...]}
            if isinstance(data_array, dict) and "articles" in data_array:
                data_array = data_array["articles"]

            # If model returned a single object, wrap it
            if isinstance(data_array, dict):
                data_array = [data_array]

            if not isinstance(data_array, list):
                logger.error("LLM response is not a JSON array")
                return [None] * len(articles)

            # Match responses to articles by URL; if mismatched, fall back to positional order
            # Normalize URLs for matching (strip whitespace and trailing slashes)
            url_to_index = {article.url.strip().rstrip("/"): idx for idx, article in enumerate(articles)}
            results: list[Dict[str, str] | None] = [None] * len(articles)
            unmatched_items = []

            for item in data_array:
                url = (item.get("url") or "").strip().rstrip("/")
                tags = self._clean_tags(item.get("tags"))
                if url in url_to_index:
                    idx = url_to_index[url]
                    results[idx] = {
                        "url": url,
                        "is_technical": item.get("is_technical", False),
                        "title_ko": item.get("title_ko", "")[:100],
                        "summary_ko": item.get("summary_ko", ""),  # No length limit - full markdown summary
                        "category": item.get("category", "OTHER"),
                        "tags": tags,
                    }
                else:
                    item["__clean_tags__"] = tags
                    unmatched_items.append(item)

            # Fill remaining slots in order for unmatched items (LLM sometimes tweaks URLs)
            remaining_indices = [i for i, val in enumerate(results) if val is None]
            for item, idx in zip(unmatched_items, remaining_indices):
                url = (item.get("url") or "").strip()
                tags = item.get("__clean_tags__") or self._clean_tags(item.get("tags"))
                logger.warning(f"Could not match URL from LLM response, assigning by order: {url}")
                results[idx] = {
                    "url": url,
                    "is_technical": item.get("is_technical", False),
                    "title_ko": item.get("title_ko", "")[:100],
                    "summary_ko": item.get("summary_ko", ""),  # No length limit - full markdown summary
                    "category": item.get("category", "OTHER"),
                    "tags": tags,
                }

            return results

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM batch response as JSON: {e}")
            logger.error(f"Problematic content (first 1000 chars): {content[:1000]}")
            logger.error(f"Problematic content (last 500 chars): {content[-500:]}")
            # Return None for all articles in this batch
            return [None] * len(articles)
        except Exception as e:
            logger.error(f"Unexpected error parsing batch response: {e}")
            return [None] * len(articles)

    # Structured output schema for OpenAI — constrained decoding forces complete JSON
    OPENAI_RESPONSE_SCHEMA = {
        "type": "json_schema",
        "json_schema": {
            "name": "article_summaries",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "articles": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "url": {"type": "string"},
                                "is_technical": {"type": "boolean"},
                                "title_ko": {"type": "string"},
                                "summary_ko": {"type": "string"},
                                "category": {
                                    "type": "string",
                                    "enum": [
                                        "AI_LLM", "DEVOPS_SRE", "INFRA_CLOUD", "DATABASE",
                                        "BLOCKCHAIN", "SECURITY", "DATA_SCIENCE", "ARCHITECTURE",
                                        "MOBILE", "FRONTEND", "BACKEND", "OTHER"
                                    ]
                                },
                                "tags": {"type": "array", "items": {"type": "string"}}
                            },
                            "required": ["url", "is_technical", "title_ko", "summary_ko", "category", "tags"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["articles"],
                "additionalProperties": False
            }
        }
    }

    async def _summarize_batch_llm(self, articles: list[RawArticle], max_tokens_override: int = None) -> list[Dict[str, str]]:
        """Call LLM once with multiple articles"""
        try:
            prompt = self._build_batch_prompt(articles)
            tokens = max_tokens_override or self.max_tokens

            response = await self.openai_client.chat.completions.create(
                model="gpt-5-nano",
                messages=[
                    {"role": "system", "content": self.SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=min(tokens, 128000),
                reasoning_effort="low",
                response_format=self.OPENAI_RESPONSE_SCHEMA
            )
            content = response.choices[0].message.content
            logger.info(
                f"OpenAI response: finish_reason={response.choices[0].finish_reason}, "
                f"usage={response.usage}"
            )

            return self._parse_batch_response(content, articles)

        except Exception as e:
            if self._is_quota_error(e):
                logger.error(f"LLM quota exceeded: {e}")
                raise LLMQuotaExceeded(str(e))
            logger.error(f"Failed to batch summarize {len(articles)} articles: {e}")
            return [None] * len(articles)

    async def summarize_batch(
        self,
        articles: list[RawArticle],
        batch_size: int = 5,
        delay: float = 5.0,
        max_tokens_override: int = None
    ) -> list[Dict[str, str]]:
        """
        Summarize multiple articles efficiently by batching them into single LLM requests

        NOTE: Uses FULL article content and generates comprehensive markdown summaries
        With OpenAI structured outputs (constrained decoding), 5 articles per batch is safe
        within the 128k output token limit (typical: ~9k tokens per long article)

        Args:
            articles: List of RawArticles to summarize
            batch_size: Number of articles per LLM request (default: 2, reduced for stability)
            delay: Delay between LLM requests in seconds (default: 5)

        Returns:
            List of summaries (or None for failed articles)
        """
        summaries = []

        for i in range(0, len(articles), batch_size):
            batch = articles[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}: {len(batch)} articles")

            # Send all articles in batch to LLM in one request
            try:
                batch_summaries = await self._summarize_batch_llm(batch, max_tokens_override=max_tokens_override)
            except LLMQuotaExceeded:
                summaries.extend([None] * len(batch))
                remaining = len(articles) - (i + len(batch))
                if remaining > 0:
                    summaries.extend([None] * remaining)
                logger.error("Aborting remaining batches due to LLM quota exhaustion")
                break
            summaries.extend(batch_summaries)

            # Delay between batches to respect rate limits
            if i + batch_size < len(articles):
                await asyncio.sleep(delay)

        return summaries

    @staticmethod
    def _fix_json_string(json_str: str) -> str:
        """
        Fix literal newlines and unescaped quotes in JSON string values

        The LLM sometimes generates JSON with:
        1. Literal newlines instead of \\n
        2. Unescaped quotes within string values
        This function uses a state machine to properly escape them
        """
        result = []
        in_string = False
        in_value = False  # True if we're in a string value (after a colon), not a key
        escape_next = False
        i = 0

        while i < len(json_str):
            char = json_str[i]

            # Handle escape sequences
            if escape_next:
                result.append(char)
                escape_next = False
                i += 1
                continue

            if char == '\\':
                result.append(char)
                escape_next = True
                i += 1
                continue

            # Handle quote characters
            if char == '"':
                # If we're in a value string, check if this quote ends the string or should be escaped
                if in_string and in_value:
                    # Look ahead to see what follows this quote
                    # Skip whitespace
                    j = i + 1
                    while j < len(json_str) and json_str[j] in ' \t\n\r':
                        j += 1

                    # If followed by comma, closing brace, or closing bracket, it's the end of the string
                    if j < len(json_str) and json_str[j] in ',}]':
                        # This is the closing quote
                        result.append(char)
                        in_string = False
                        in_value = False
                    else:
                        # This is an unescaped quote within the content - escape it
                        result.append('\\')
                        result.append(char)
                    i += 1
                    continue

                # Regular quote handling (entering/exiting strings for keys)
                result.append(char)
                if in_string:
                    # Exiting a string
                    in_string = False
                    in_value = False
                else:
                    # Entering a string - determine if it's a key or value
                    # Look backwards to see if we're after a colon (value) or not (key)
                    # Find the last non-whitespace character
                    j = len(result) - 2
                    while j >= 0 and result[j] in ' \t\n\r':
                        j -= 1

                    in_string = True
                    in_value = (j >= 0 and result[j] == ':')

                i += 1
                continue

            # If we're in a value string, escape literal newlines
            if in_string and in_value:
                if char == '\n':
                    result.append('\\n')
                    i += 1
                    continue
                elif char == '\r':
                    result.append('\\r')
                    i += 1
                    continue
                elif char == '\t':
                    result.append('\\t')
                    i += 1
                    continue

            result.append(char)
            i += 1

        fixed = ''.join(result)
        if fixed != json_str:
            logger.debug("JSON string fixed - escaped literal newlines and unescaped quotes in values")
        return fixed

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """Remove wrapping markdown code fences if present."""
        text = text.strip()
        if text.startswith("```"):
            text = text[3:]
            if text.startswith("json"):
                text = text[4:]
            if "```" in text:
                text = text.split("```")[0]
        return text.strip()

    @staticmethod
    def _extract_json_payload(text: str, expect_array: bool) -> str:
        """Extract the JSON array/object substring if response includes extra text."""
        text = text.strip()
        if expect_array:
            start = text.find("[")
            if start == -1:
                return text
            end = SummarizerService._find_matching_bracket(text, start, "[", "]")
        else:
            start = text.find("{")
            if start == -1:
                return text
            end = SummarizerService._find_matching_bracket(text, start, "{", "}")
        if end != -1 and end > start:
            return text[start:end + 1]
        # If no matching end, return from start to allow repair attempts
        return text[start:]

    @staticmethod
    def _find_matching_bracket(text: str, start: int, open_char: str, close_char: str) -> int:
        """Find matching closing bracket for a JSON array/object, ignoring brackets in strings."""
        in_string = False
        escape_next = False
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if escape_next:
                escape_next = False
                continue
            if ch == "\\":
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == open_char:
                depth += 1
            elif ch == close_char:
                depth -= 1
                if depth == 0:
                    return i
        return -1

    @staticmethod
    def _remove_trailing_commas(text: str) -> str:
        """Remove trailing commas before closing braces/brackets."""
        return re.sub(r",\s*([}\]])", r"\1", text)

    @staticmethod
    def _close_unterminated_json(text: str, expect_array: bool) -> str:
        """
        Close unterminated strings and brackets/braces to salvage truncated JSON.
        """
        s = text.strip()
        if expect_array and not s.lstrip().startswith("["):
            # If it looks like a single object, wrap it in an array
            if "{" in s:
                s = "[" + s
            else:
                s = "[" + s

        in_string = False
        escape_next = False
        stack: list[str] = []

        for ch in s:
            if escape_next:
                escape_next = False
                continue
            if ch == "\\":
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch in "{[":
                stack.append(ch)
            elif ch in "}]":
                if stack:
                    stack.pop()

        if in_string:
            s += '"'

        while stack:
            opener = stack.pop()
            s += "}" if opener == "{" else "]"

        if expect_array and not s.rstrip().endswith("]"):
            s += "]"
        return s

    @staticmethod
    def _trim_to_last_complete_object(text: str) -> Optional[str]:
        """
        Trim JSON array to the last complete top-level object.
        Useful when output is truncated mid-object.
        """
        s = text.strip()
        in_string = False
        escape_next = False
        bracket_depth = 0
        brace_depth = 0
        last_obj_end = None

        for i, ch in enumerate(s):
            if escape_next:
                escape_next = False
                continue
            if ch == "\\":
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "[":
                bracket_depth += 1
            elif ch == "]":
                bracket_depth = max(0, bracket_depth - 1)
            elif ch == "{":
                brace_depth += 1
            elif ch == "}":
                brace_depth = max(0, brace_depth - 1)
                if brace_depth == 0 and bracket_depth >= 1:
                    last_obj_end = i

        if last_obj_end is None:
            return None

        start = s.find("[")
        if start == -1:
            return None
        trimmed = s[start:last_obj_end + 1]
        return trimmed + "]"

    def _safe_json_loads(self, content: str, expect_array: bool):
        """Best-effort JSON parsing with multiple repair attempts."""
        original = content
        content = self._strip_code_fences(content)
        content = self._extract_json_payload(content, expect_array=expect_array)

        candidates: list[tuple[str, str]] = []

        # If we expect an array but got a single object, try wrapping
        if expect_array:
            stripped = content.strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                candidates.append(("wrapped_single_object", f"[{content}]"))

        candidates.append(("raw", content))
        candidates.append(("fixed_strings", self._fix_json_string(content)))

        for label, candidate in candidates:
            candidate = self._remove_trailing_commas(candidate)
            try:
                data = json.loads(candidate)
                if label != "raw":
                    logger.info(f"Parsed JSON after repair: {label}")
                return data
            except json.JSONDecodeError as e:
                if label == "raw":
                    logger.warning(f"Initial JSON parse failed at line {e.lineno}, col {e.colno}: {e.msg}")
                    logger.debug(f"Error context: {candidate[max(0, e.pos-100):e.pos+100]}")
                continue

        # Attempt to close unterminated JSON (truncation)
        repaired = self._close_unterminated_json(self._fix_json_string(content), expect_array=expect_array)
        repaired = self._remove_trailing_commas(repaired)
        try:
            data = json.loads(repaired)
            logger.info("Parsed JSON after closing unterminated structures")
            return data
        except json.JSONDecodeError:
            pass

        # Final attempt: trim to last complete object
        if expect_array:
            trimmed = self._trim_to_last_complete_object(repaired)
            if trimmed:
                trimmed = self._remove_trailing_commas(trimmed)
                try:
                    data = json.loads(trimmed)
                    logger.warning("Parsed JSON by trimming to last complete object (truncated output)")
                    return data
                except json.JSONDecodeError:
                    pass

        logger.error("All JSON repair attempts failed")
        logger.error(f"Raw content length: {len(original)} chars")
        return None

    @staticmethod
    def _is_quota_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return "quota" in msg and "exceed" in msg

    @staticmethod
    def _clean_tags(tags) -> list[str]:
        """Normalize tags list to <=5 lowercase strings without spaces."""
        if not tags:
            return []
        if isinstance(tags, str):
            tags = [tags]
        cleaned = []
        for t in tags:
            if not isinstance(t, str):
                continue
            tag = t.strip().lower().replace(" ", "-")
            if tag:
                cleaned.append(tag)
        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for tag in cleaned:
            if tag in seen:
                continue
            seen.add(tag)
            deduped.append(tag)
        return deduped[:5]
