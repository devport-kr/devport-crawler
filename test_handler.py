#!/usr/bin/env python3
"""
Test script for Lambda handler

Run this script to validate that the handler works correctly without
needing to deploy to AWS Lambda.

Usage:
    python test_handler.py
"""

import sys
from app.handler import lambda_handler


def test_event(source: str, description: str):
    """Test a single event payload"""
    print(f"\n{'=' * 70}")
    print(f"Testing: {description}")
    print(f"Event: {{'source': '{source}'}}")
    print('=' * 70)

    event = {"source": source}

    try:
        result = lambda_handler(event, None)
        print(f"\n✅ Status: {result.get('statusCode')}")
        print(f"Source: {result.get('source')}")

        if result.get('statusCode') == 200:
            stats = result.get('result', {})
            print(f"\nResults:")
            if isinstance(stats, dict):
                for key, value in stats.items():
                    print(f"  {key}: {value}")
            else:
                print(f"  {stats}")
        else:
            print(f"\n❌ Error: {result.get('error')}")

    except Exception as e:
        print(f"\n❌ Exception: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests"""
    print("=" * 70)
    print("DevPort Crawler - Lambda Handler Test Suite")
    print("=" * 70)
    print("\nThis will test the Lambda handler with different event payloads.")
    print("NOTE: This requires database and API credentials to be configured.")
    print("\nPress Ctrl+C to stop at any time.\n")

    # Test cases
    tests = [
        ("github", "GitHub Trending Crawler"),
        ("hashnode", "Hashnode Crawler"),
        ("medium", "Medium Crawler"),
        ("reddit", "Reddit Crawler"),
        ("llm_rankings", "LLM Rankings Crawler"),
        ("all_blogs", "All Blog Crawlers"),
    ]

    # Ask user which test to run
    print("Available tests:")
    for i, (source, desc) in enumerate(tests, 1):
        print(f"  {i}. {desc} (source={source})")
    print("  7. Run all tests")
    print("  0. Exit")

    try:
        choice = input("\nEnter test number (default: 1): ").strip()
        if not choice:
            choice = "1"

        choice_num = int(choice)

        if choice_num == 0:
            print("Exiting.")
            return

        if choice_num == 7:
            # Run all tests
            for source, desc in tests:
                test_event(source, desc)
                print("\n" + "-" * 70)
        elif 1 <= choice_num <= 6:
            # Run single test
            source, desc = tests[choice_num - 1]
            test_event(source, desc)
        else:
            print("Invalid choice.")
            return

        print("\n" + "=" * 70)
        print("Testing complete!")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except ValueError:
        print("Invalid input. Please enter a number.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
