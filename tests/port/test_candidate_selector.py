from datetime import UTC, datetime, timedelta

from app.services.port.candidate_selector import CandidateSelector, RepoCandidate, SelectionConfig


def _repo(
    external_id: str,
    full_name: str,
    *,
    stars: int,
    days_ago: int,
    description: str = "",
    topics: tuple[str, ...] = (),
) -> RepoCandidate:
    return RepoCandidate(
        external_id=external_id,
        full_name=full_name,
        stars=stars,
        pushed_at=datetime.now(UTC) - timedelta(days=days_ago),
        description=description,
        topics=topics,
    )


def test_selector_preserves_manual_baseline_then_expands() -> None:
    selector = CandidateSelector(
        SelectionConfig(global_target=3, diversity_soft_cap=2, now_provider=datetime.now)
    )

    manual = [_repo("manual-1", "baseline/core", stars=12, days_ago=40, topics=("python",))]
    auto = [
        _repo("manual-1", "baseline/core", stars=1000, days_ago=1, topics=("python",)),
        _repo("auto-1", "new-org/new-toolkit", stars=350, days_ago=3, topics=("python", "crawler")),
        _repo("auto-2", "new-org/noise", stars=700, days_ago=2, topics=("gaming",)),
    ]

    selected = selector.select_candidates(
        manual_baseline=manual,
        auto_candidates=auto,
        relevance_keywords={"python", "crawler"},
        target_count=3,
    )

    assert [candidate.external_id for candidate in selected] == ["manual-1", "auto-1"]


def test_selector_allows_underfill_when_relevant_candidates_are_insufficient() -> None:
    selector = CandidateSelector(
        SelectionConfig(global_target=5, diversity_soft_cap=2, now_provider=datetime.now)
    )

    selected = selector.select_candidates(
        manual_baseline=[],
        auto_candidates=[
            _repo("good-1", "alpha/relevant", stars=50, days_ago=8, topics=("kubernetes",)),
            _repo("bad-1", "beta/offtopic", stars=999, days_ago=1, topics=("design",)),
        ],
        relevance_keywords={"kubernetes"},
        target_count=5,
    )

    assert len(selected) == 1
    assert selected[0].external_id == "good-1"


def test_selector_uses_soft_diversity_preference_without_hard_organization_limit() -> None:
    selector = CandidateSelector(
        SelectionConfig(
            global_target=3,
            diversity_soft_cap=1,
            diversity_penalty=0.45,
            now_provider=datetime.now,
        )
    )

    selected = selector.select_candidates(
        manual_baseline=[],
        auto_candidates=[
            _repo("a1", "org-a/repo-one", stars=700, days_ago=2, topics=("llm", "serving")),
            _repo("a2", "org-a/repo-two", stars=650, days_ago=2, topics=("llm", "serving")),
            _repo("b1", "org-b/repo-one", stars=520, days_ago=4, topics=("llm", "serving")),
        ],
        relevance_keywords={"llm", "serving"},
        target_count=3,
    )

    assert [candidate.external_id for candidate in selected] == ["a1", "b1", "a2"]
