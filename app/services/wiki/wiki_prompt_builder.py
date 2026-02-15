"""Prompt builder for wiki section generation with adaptive depth."""

from __future__ import annotations

import logging

from openai import AsyncOpenAI

from app.config.settings import settings

logger = logging.getLogger(__name__)


class WikiPromptBuilder:
    """Builds prompts for LLM-based wiki content generation with actual OpenAI calls."""

    def __init__(self) -> None:
        self._client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    async def build_what_explanation(
        self,
        *,
        name: str,
        description: str,
        topics: list[str] | None = None,
    ) -> str:
        """Generate 'what' section deep dive content using LLM.
        
        Args:
            name: Project name.
            description: Project description.
            topics: Repository topics/tags.
            
        Returns:
            LLM-generated markdown content for deep dive section.
        """
        topics_list = topics or []
        topics_text = ", ".join(topics_list[:5]) if topics_list else "general software development"

        prompt = f"""You are a technical documentation writer creating a "What is this project?" section for an open-source repository wiki.

Project: {name}
Description: {description}
Topics: {topics_text}

Write a deep technical explanation covering:

1. **Purpose and Domain** - What problem this project solves and what domain it operates in
2. **Core Capabilities** - What it actually does (be specific about features/functionality)
3. **Target Users** - Who should use this (specific personas: backend devs, DevOps engineers, data scientists, etc.)
4. **Why This Project Matters** - Its position in the ecosystem, unique value proposition

Write in engineer-to-engineer tone. Be factual and specific. Use markdown with headers (###). 
Do NOT include generic phrases like "to be populated" or "based on context". 
Length: 200-400 words."""

        try:
            response = await self._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a technical documentation writer for open-source projects. Write clear, factual, engineer-to-engineer content."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=600,
            )
            
            content = response.choices[0].message.content or ""
            logger.info(f"Generated 'what' section for {name} ({len(content)} chars)")
            return content.strip()
            
        except Exception as e:
            logger.warning(f"LLM generation failed for {name} 'what' section: {e}")
            # Fallback to template
            return f"""### Purpose and Domain

{description}

### Core Capabilities

{name} provides functionality in the {topics_text} domain.

### Target Users

Engineers and teams working with {topics_text} technologies."""

    async def build_how_explanation(
        self,
        *,
        name: str,
        description: str,
    ) -> str:
        """Generate 'how it works' section using LLM.
        
        Args:
            name: Project name.
            description: Project description.
            
        Returns:
            LLM-generated markdown explaining key concepts and workflow.
        """
        prompt = f"""You are a technical documentation writer creating a "How it works" section for an open-source repository wiki.

Project: {name}
Description: {description}

Write a deep technical explanation covering:

1. **Key Concepts** - Core abstractions, patterns, or mental models users need to understand
2. **Workflow** - Typical usage flow from setup to execution (be specific about steps)
3. **Integration Patterns** - How this fits into a developer's stack/workflow

Write in engineer-to-engineer tone. Be specific and actionable. Use markdown with headers (###).
Do NOT use generic placeholder text. Infer realistic workflow based on the project type.
Length: 200-400 words."""

        try:
            response = await self._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a technical documentation writer for open-source projects. Write clear, factual, actionable content."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=600,
            )
            
            content = response.choices[0].message.content or ""
            logger.info(f"Generated 'how' section for {name} ({len(content)} chars)")
            return content.strip()
            
        except Exception as e:
            logger.warning(f"LLM generation failed for {name} 'how' section: {e}")
            return f"""### Key Concepts

{description}

### Workflow

1. Install and configure {name}
2. Integrate into your application
3. Use the provided APIs for your use case"""

    async def build_architecture_explanation(
        self,
        *,
        name: str,
        language: str,
    ) -> str:
        """Generate architecture/codebase explanation using LLM.
        
        Args:
            name: Project name.
            language: Primary programming language.
            
        Returns:
            LLM-generated markdown explaining architecture.
        """
        prompt = f"""You are a technical documentation writer creating an "Architecture and Codebase" section for an open-source repository wiki.

Project: {name}
Primary Language: {language}

Write a deep technical explanation covering:

1. **Technical Stack** - Languages, frameworks, key dependencies (infer realistic ones for a {language} project)
2. **Architectural Pattern** - Overall design approach (monolith, microservices, library, CLI tool, etc.)
3. **Core Components** - Main modules/packages and their responsibilities
4. **Design Patterns** - Key patterns used (if applicable to project type)

Write in engineer-to-engineer tone. Be specific about technical choices. Use markdown with headers (###).
Infer realistic architecture based on project type and language. Be concrete, not generic.
Length: 200-400 words."""

        try:
            response = await self._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a technical documentation writer for open-source projects. Write clear, specific architectural explanations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=600,
            )
            
            content = response.choices[0].message.content or ""
            logger.info(f"Generated 'architecture' section for {name} ({len(content)} chars)")
            return content.strip()
            
        except Exception as e:
            logger.warning(f"LLM generation failed for {name} 'architecture' section: {e}")
            return f"""### Technical Stack

- **Primary Language**: {language}
- Standard {language} conventions and tooling

### Architecture

Modular design with clear component separation."""
