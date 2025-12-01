"""Korean summarization service using LLM APIs"""

from typing import Dict, Optional
import asyncio
import json
import logging
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
import google.generativeai as genai
from app.config.settings import settings
from app.crawlers.base import RawArticle

logger = logging.getLogger(__name__)


class SummarizerService:
    """Service to generate Korean summaries using LLM APIs"""

    def __init__(self):
        self.provider = settings.LLM_PROVIDER

        if self.provider == "openai":
            if not settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER is 'openai'")
            self.openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        elif self.provider == "anthropic":
            if not settings.ANTHROPIC_API_KEY:
                raise ValueError("ANTHROPIC_API_KEY is required when LLM_PROVIDER is 'anthropic'")
            self.anthropic_client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        elif self.provider == "gemini":
            if not settings.GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY is required when LLM_PROVIDER is 'gemini'")
            genai.configure(api_key=settings.GEMINI_API_KEY)
            self.gemini_model = genai.GenerativeModel('gemini-2.5-flash-lite')
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    async def summarize(self, article: RawArticle) -> Dict[str, str]:
        """
        Generate Korean summary and categorization for an article

        Args:
            article: RawArticle to summarize

        Returns:
            Dictionary with 'title_ko', 'summary_ko', and 'category' keys, or None if failed
        """
        try:
            if self.provider == "openai":
                return await self._summarize_openai(article)
            elif self.provider == "anthropic":
                return await self._summarize_anthropic(article)
            elif self.provider == "gemini":
                return await self._summarize_gemini(article)
        except Exception as e:
            logger.error(f"Failed to summarize article: {e}")
            # Return None to indicate failure - don't save articles without proper summarization
            return None

    async def _summarize_openai(self, article: RawArticle) -> Dict[str, str]:
        """Generate summary using OpenAI GPT"""
        prompt = self._build_prompt(article)

        response = await self.openai_client.chat.completions.create(
            model="gpt-4o-mini",  # Use mini for cost savings
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that summarizes developer content in Korean."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=500
        )

        content = response.choices[0].message.content
        return self._parse_response(content)

    async def _summarize_anthropic(self, article: RawArticle) -> Dict[str, str]:
        """Generate summary using Anthropic Claude"""
        prompt = self._build_prompt(article)

        response = await self.anthropic_client.messages.create(
            model="claude-3-5-haiku-20241022",  # Use Haiku for cost savings
            max_tokens=500,
            temperature=0.3,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        content = response.content[0].text
        return self._parse_response(content)

    async def _summarize_gemini(self, article: RawArticle) -> Dict[str, str]:
        """Generate summary using Google Gemini"""
        prompt = self._build_prompt(article)

        # Gemini API is sync, so we need to run it in executor
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=500,
                )
            )
        )

        content = response.text
        return self._parse_response(content)

    def _build_batch_prompt(self, articles: list[RawArticle]) -> str:
        """Build prompt for multiple articles at once"""
        articles_text = ""
        for i, article in enumerate(articles, 1):
            content_preview = (article.content or "")[:500]  # Shorter preview for batching
            articles_text += f"""
Article {i}:
Title: {article.title_en}
URL: {article.url}
Tags: {', '.join(article.tags[:10])}
Content: {content_preview}

"""

        prompt = f"""Summarize and categorize these {len(articles)} developer articles in Korean.

{articles_text}

For EACH article, provide a JSON object with:
1. "url": The article URL (to match it back)
2. "title_ko": A concise Korean title (max 100 characters), that represents the original english title at best
3. "summary_ko": A detailed Korean summary (2-3 sentences, max 300 characters) that best reflects the content of the article,
    also that attacts users to read
4. "category": One of these categories:
   - AI_LLM, DEVOPS_SRE, INFRA_CLOUD, DATABASE, BLOCKCHAIN, SECURITY,
   - DATA_SCIENCE, ARCHITECTURE, MOBILE, FRONTEND, BACKEND, OTHER

Return a JSON array with {len(articles)} objects in the SAME ORDER as the articles above.

Response format:
[
  {{"url": "...", "title_ko": "...", "summary_ko": "...", "category": "AI_LLM"}},
  {{"url": "...", "title_ko": "...", "summary_ko": "...", "category": "BACKEND"}},
  ...
]"""
        return prompt

    def _build_prompt(self, article: RawArticle) -> str:
        """Build the prompt for single article (legacy, keeping for compatibility)"""
        content_preview = (article.content or "")[:1000]  # Limit content length

        prompt = f"""Summarize and categorize this developer article in Korean:

Title: {article.title_en}
Tags: {', '.join(article.tags[:10])}
Content: {content_preview}

Provide a JSON response with:
1. "title_ko": A concise Korean title (max 100 characters)
2. "summary_ko": A detailed Korean summary (2-3 sentences, max 300 characters)
3. "category": One of the following categories that best matches this article:
   - AI_LLM (AI, LLM, GPT, machine learning, deep learning)
   - DEVOPS_SRE (DevOps, SRE, CI/CD, monitoring, kubernetes, docker)
   - INFRA_CLOUD (AWS, Azure, GCP, cloud infrastructure, serverless)
   - DATABASE (SQL, NoSQL, PostgreSQL, MongoDB, database design)
   - BLOCKCHAIN (Blockchain, crypto, Ethereum, Web3, smart contracts)
   - SECURITY (Security, authentication, encryption, vulnerabilities)
   - DATA_SCIENCE (Data science, analytics, visualization, big data)
   - ARCHITECTURE (System architecture, microservices, design patterns)
   - MOBILE (iOS, Android, mobile app development)
   - FRONTEND (React, Vue, Angular, JavaScript, CSS, web frontend)
   - BACKEND (Backend, API, Node.js, Python, Java, server-side)
   - OTHER (anything else that doesn't fit above categories)

The summary should:
- Explain what the article is about
- Mention key technologies or concepts
- Be informative and engaging for Korean developers

Response format:
{{
  "title_ko": "...",
  "summary_ko": "...",
  "category": "AI_LLM"
}}"""

        return prompt

    def _parse_batch_response(self, content: str, articles: list[RawArticle]) -> list[Dict[str, str]]:
        """Parse LLM batch response into list of structured data"""
        try:
            # Try to extract JSON from the response
            content = content.strip()

            # Remove markdown code blocks if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            data_array = json.loads(content)

            if not isinstance(data_array, list):
                logger.error("LLM response is not a JSON array")
                return [None] * len(articles)

            # Match responses to articles by URL
            url_to_article = {article.url: article for article in articles}
            results = []

            for item in data_array:
                url = item.get("url", "")
                if url in url_to_article:
                    results.append({
                        "url": url,
                        "title_ko": item.get("title_ko", "")[:100],
                        "summary_ko": item.get("summary_ko", "")[:300],
                        "category": item.get("category", "OTHER")
                    })
                else:
                    logger.warning(f"Could not match URL from LLM response: {url}")
                    results.append(None)

            # Fill in None for any missing articles
            while len(results) < len(articles):
                results.append(None)

            return results

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM batch response as JSON: {e}")
            return [None] * len(articles)

    def _parse_response(self, content: str) -> Dict[str, str]:
        """Parse LLM response into structured data (single article)"""
        try:
            # Try to extract JSON from the response
            content = content.strip()

            # Remove markdown code blocks if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            data = json.loads(content)

            return {
                "title_ko": data.get("title_ko", "")[:100],
                "summary_ko": data.get("summary_ko", "")[:300],
                "category": data.get("category", "OTHER")
            }
        except json.JSONDecodeError:
            logger.error(f"Failed to parse LLM response as JSON: {content}")
            # Return None to indicate failure - don't save articles with failed summarization
            return None

    async def _summarize_batch_llm(self, articles: list[RawArticle]) -> list[Dict[str, str]]:
        """Call LLM once with multiple articles"""
        try:
            prompt = self._build_batch_prompt(articles)

            if self.provider == "openai":
                response = await self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that summarizes developer content in Korean."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=2000  # More tokens for batch response
                )
                content = response.choices[0].message.content

            elif self.provider == "anthropic":
                response = await self.anthropic_client.messages.create(
                    model="claude-3-5-haiku-20241022",
                    max_tokens=2000,
                    temperature=0.3,
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.content[0].text

            elif self.provider == "gemini":
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.gemini_model.generate_content(
                        prompt,
                        generation_config=genai.types.GenerationConfig(
                            temperature=0.3,
                            max_output_tokens=10000,  # Increased to 10k for 25-article batches
                        )
                    )
                )
                content = response.text
                logger.debug(f"Gemini response length: {len(content)} chars")

            return self._parse_batch_response(content, articles)

        except Exception as e:
            logger.error(f"Failed to batch summarize {len(articles)} articles: {e}")
            return [None] * len(articles)

    async def summarize_batch(
        self,
        articles: list[RawArticle],
        batch_size: int = 25,
        delay: float = 3.0
    ) -> list[Dict[str, str]]:
        """
        Summarize multiple articles efficiently by batching them into single LLM requests

        Args:
            articles: List of RawArticles to summarize
            batch_size: Number of articles per LLM request (default: 25)
            delay: Delay between LLM requests in seconds (default: 10)

        Returns:
            List of summaries (or None for failed articles)
        """
        summaries = []

        for i in range(0, len(articles), batch_size):
            batch = articles[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}: {len(batch)} articles")

            # Send all articles in batch to LLM in one request
            batch_summaries = await self._summarize_batch_llm(batch)
            summaries.extend(batch_summaries)

            # Delay between batches to respect rate limits
            if i + batch_size < len(articles):
                await asyncio.sleep(delay)

        return summaries
