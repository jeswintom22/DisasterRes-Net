"""Step 0: agentic disaster monitoring and Twitter image collection."""

from agents.coordinator_agent import CoordinatorAgent


def main() -> int:
    print("=" * 60)
    print("STEP 0: AGENTIC DISASTER MONITORING")
    print("=" * 60)
    CoordinatorAgent().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

