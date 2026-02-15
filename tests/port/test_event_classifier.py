from app.services.port.event_classifier import classify_event


def test_classify_event_returns_lowercase_types() -> None:
    result = classify_event(
        title="v1.2.0 release",
        body="Added new API endpoints and fixed regression bug.",
    )

    assert all(item == item.lower() for item in result.event_types)


def test_classify_event_detects_security_with_priority() -> None:
    result = classify_event(
        title="Security hotfix",
        body="Fixes CVE-2026-9999 authentication bypass vulnerability.",
    )

    assert "security" in result.event_types
    assert result.is_security is True


def test_classify_event_falls_back_to_misc() -> None:
    result = classify_event(
        title="v0.1.0",
        body="abcdef1234567890",
    )

    assert result.event_types == ["misc"]
