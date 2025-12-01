"""Content categorization service"""

from typing import List
from app.models.article import Category
from app.crawlers.base import RawArticle
import logging

logger = logging.getLogger(__name__)


class CategorizerService:
    """Service to categorize articles based on tags and content"""

    # Keyword mappings for each category
    CATEGORY_KEYWORDS = {
        Category.AI_LLM: [
            "ai", "llm", "gpt", "claude", "gemini", "chatbot", "ml", "machine learning",
            "deep learning", "neural network", "transformer", "openai", "anthropic",
            "artificial intelligence", "chatgpt", "generative", "langchain", "rag"
        ],
        Category.DEVOPS_SRE: [
            "devops", "sre", "ci/cd", "jenkins", "github actions", "gitlab", "deployment",
            "monitoring", "prometheus", "grafana", "kubernetes", "k8s", "docker", "container",
            "observability", "reliability", "incident", "automation", "terraform", "ansible"
        ],
        Category.INFRA_CLOUD: [
            "aws", "azure", "gcp", "cloud", "serverless", "lambda", "s3", "ec2",
            "infrastructure", "iaas", "paas", "cloudflare", "cdn", "cloud computing",
            "cloud run", "app engine", "elastic", "fargate", "ecs"
        ],
        Category.DATABASE: [
            "database", "sql", "nosql", "postgresql", "mysql", "mongodb", "redis",
            "elasticsearch", "cassandra", "dynamodb", "firestore", "db", "orm",
            "query", "indexing", "migration", "prisma", "sequelize", "sqlalchemy"
        ],
        Category.BLOCKCHAIN: [
            "blockchain", "crypto", "ethereum", "bitcoin", "web3", "solidity",
            "smart contract", "nft", "defi", "dapp", "wallet", "consensus",
            "cryptocurrency", "token", "metamask"
        ],
        Category.SECURITY: [
            "security", "authentication", "authorization", "oauth", "jwt", "encryption",
            "vulnerability", "exploit", "penetration", "cybersecurity", "infosec",
            "xss", "csrf", "sql injection", "zero-day", "firewall", "vpn", "ssl", "tls"
        ],
        Category.DATA_SCIENCE: [
            "data science", "analytics", "pandas", "numpy", "jupyter", "visualization",
            "statistics", "data analysis", "big data", "data engineering", "etl",
            "pipeline", "spark", "hadoop", "airflow", "dbt", "tableau", "power bi"
        ],
        Category.ARCHITECTURE: [
            "architecture", "microservices", "monolith", "design pattern", "system design",
            "scalability", "distributed", "event-driven", "cqrs", "saga", "api gateway",
            "service mesh", "clean architecture", "hexagonal", "ddd", "domain-driven"
        ],
        Category.MOBILE: [
            "mobile", "ios", "android", "swift", "kotlin", "react native", "flutter",
            "xamarin", "app development", "mobile app", "iphone", "sdk", "xcode",
            "android studio", "objective-c", "swiftui"
        ],
        Category.FRONTEND: [
            "frontend", "react", "vue", "angular", "svelte", "next.js", "nuxt",
            "javascript", "typescript", "html", "css", "tailwind", "bootstrap",
            "webpack", "vite", "ui", "ux", "responsive", "web design", "sass"
        ],
        Category.BACKEND: [
            "backend", "api", "rest", "graphql", "node.js", "python", "java", "go",
            "rust", "spring boot", "fastapi", "express", "django", "flask", "nest.js",
            "server", "microservice", "grpc", "websocket", "kafka", "rabbitmq"
        ]
    }

    @staticmethod
    def categorize(article: RawArticle) -> Category:
        """
        Categorize an article based on its tags and content

        Args:
            article: RawArticle to categorize

        Returns:
            Category enum value
        """
        # Combine tags and title for analysis
        text_to_analyze = " ".join([
            article.title_en.lower(),
            " ".join(article.tags).lower(),
            (article.content or "").lower()[:500]  # First 500 chars of content
        ])

        # Count keyword matches for each category
        category_scores = {}

        for category, keywords in CategorizerService.CATEGORY_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in text_to_analyze)
            if score > 0:
                category_scores[category] = score

        # Return category with highest score
        if category_scores:
            best_category = max(category_scores, key=category_scores.get)
            logger.debug(
                f"Categorized '{article.title_en[:50]}' as {best_category.value} "
                f"(score: {category_scores[best_category]})"
            )
            return best_category

        # Default to OTHER if no matches
        logger.debug(f"No category match for '{article.title_en[:50]}', using OTHER")
        return Category.OTHER

    @staticmethod
    def categorize_by_language(language: str) -> Category:
        """
        Categorize by programming language (for GitHub repos)

        Args:
            language: Programming language name

        Returns:
            Category enum value
        """
        language_lower = (language or "").lower()

        # Language to category mapping
        language_map = {
            "python": Category.BACKEND,
            "javascript": Category.FRONTEND,
            "typescript": Category.FRONTEND,
            "java": Category.BACKEND,
            "go": Category.BACKEND,
            "rust": Category.BACKEND,
            "c++": Category.BACKEND,
            "c#": Category.BACKEND,
            "php": Category.BACKEND,
            "ruby": Category.BACKEND,
            "swift": Category.MOBILE,
            "kotlin": Category.MOBILE,
            "dart": Category.MOBILE,
            "solidity": Category.BLOCKCHAIN,
            "html": Category.FRONTEND,
            "css": Category.FRONTEND,
            "shell": Category.DEVOPS_SRE,
            "dockerfile": Category.DEVOPS_SRE,
        }

        for lang_key, category in language_map.items():
            if lang_key in language_lower:
                return category

        return Category.OTHER
