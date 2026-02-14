"""Default Port seed catalog for autonomous project discovery."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PortSeed:
    """Port seed definition used for initial port/project discovery."""

    slug: str
    name: str
    port_number: int
    description: str
    accent_color: str
    keywords: tuple[str, ...]
    baseline_repos: tuple[str, ...]


DEFAULT_PORT_SEEDS: tuple[PortSeed, ...] = (
    PortSeed(
        slug="llm",
        name="LLMs",
        port_number=11434,
        description="Local and hosted LLM inference and tooling ecosystem.",
        accent_color="#7c3aed",
        keywords=("llm", "inference", "model serving", "rag"),
        baseline_repos=(
            "ollama/ollama",
            "vllm-project/vllm",
            "huggingface/text-generation-inference",
        ),
    ),
    PortSeed(
        slug="kubernetes",
        name="Kubernetes",
        port_number=6443,
        description="Container orchestration and cloud-native cluster runtime.",
        accent_color="#2563eb",
        keywords=("kubernetes", "k8s", "cluster", "operator"),
        baseline_repos=(
            "kubernetes/kubernetes",
            "kubernetes/minikube",
            "helm/helm",
        ),
    ),
    PortSeed(
        slug="devops",
        name="DevOps/CI",
        port_number=8080,
        description="Build, deploy, and delivery automation workflows.",
        accent_color="#ef4444",
        keywords=("devops", "ci", "cd", "deployment pipeline"),
        baseline_repos=(
            "argoproj/argo-cd",
            "jenkinsci/jenkins",
            "actions/runner",
        ),
    ),
    PortSeed(
        slug="docker",
        name="Docker",
        port_number=2375,
        description="Container engine, packaging, and runtime distribution.",
        accent_color="#0ea5e9",
        keywords=("docker", "container", "image", "container runtime"),
        baseline_repos=(
            "moby/moby",
            "docker/compose",
            "containerd/containerd",
        ),
    ),
    PortSeed(
        slug="database",
        name="Databases",
        port_number=5432,
        description="Transactional and analytical database engines and tooling.",
        accent_color="#f59e0b",
        keywords=("database", "sql", "postgres", "query engine"),
        baseline_repos=(
            "postgres/postgres",
            "mysql/mysql-server",
            "redis/redis",
        ),
    ),
    PortSeed(
        slug="monitoring",
        name="Monitoring",
        port_number=9090,
        description="Observability stack for metrics, logs, traces, and alerts.",
        accent_color="#22c55e",
        keywords=("monitoring", "observability", "metrics", "prometheus"),
        baseline_repos=(
            "prometheus/prometheus",
            "grafana/grafana",
            "open-telemetry/opentelemetry-collector",
        ),
    ),
)


def get_default_port_seeds() -> list[PortSeed]:
    """Return a mutable list of default seed definitions."""

    return list(DEFAULT_PORT_SEEDS)
