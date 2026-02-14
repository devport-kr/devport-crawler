from __future__ import annotations

import asyncio
from dataclasses import dataclass

from app.crawlers.port.contracts import FetchResult, FetchState
from app.services.port.overview_sources import OverviewSourceAggregator


@dataclass
class FakeGitHubContentClient:
    contents: dict[str, str]

    async def get_content(self, _owner: str, _repo: str, path: str) -> FetchResult[str]:
        if path not in self.contents:
            return FetchResult(state=FetchState.EMPTY, data="")
        return FetchResult(state=FetchState.OK, data=self.contents[path])


def test_collect_readme_only_source_payload() -> None:
    client = FakeGitHubContentClient(
        contents={
            "README.md": "# Demo\n\n## Overview\nA concise project overview.\n\n## Usage\nRun the service.",
        }
    )
    aggregator = OverviewSourceAggregator(client)

    payload = asyncio.run(aggregator.collect(owner="acme", repo="demo"))

    assert payload.skipped is False
    assert payload.raw_hash
    assert payload.source_url.endswith("/README.md")
    assert "SOURCE: README" in payload.raw_text
    assert "project overview" in payload.raw_text.lower()


def test_collect_readme_and_docs_with_deduped_blocks() -> None:
    readme = (
        "# Demo\n\n"
        "## Quickstart\nInstall and run.\n\n"
        "See [docs](docs/guide.md) and [model card](https://github.com/acme/demo/blob/main/docs/model.md)."
    )
    client = FakeGitHubContentClient(
        contents={
            "README.md": readme,
            "docs/guide.md": "# Guide\n\n## Usage\nInstall and run.",
            "docs/model.md": "# Model\n\n## Architecture\nEncoder + decoder pipeline.",
        }
    )
    aggregator = OverviewSourceAggregator(client)

    payload = asyncio.run(aggregator.collect(owner="acme", repo="demo"))

    assert "docs/guide.md" in payload.raw_text
    assert "docs/model.md" in payload.raw_text
    assert "Encoder + decoder pipeline" in payload.raw_text
    assert len(payload.links) >= 2


def test_collect_marks_unchanged_hash_as_skipped() -> None:
    client = FakeGitHubContentClient(
        contents={
            "README.md": "# Demo\n\n## Intro\nStable content.",
        }
    )
    aggregator = OverviewSourceAggregator(client)

    first = asyncio.run(aggregator.collect(owner="acme", repo="demo"))
    second = asyncio.run(aggregator.collect(owner="acme", repo="demo", previous_raw_hash=first.raw_hash))

    assert second.raw_hash == first.raw_hash
    assert second.skipped is True
