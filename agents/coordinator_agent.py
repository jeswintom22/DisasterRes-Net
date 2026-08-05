"""CoordinatorAgent with an explicit ReAct-style reasoning loop."""

import os
from typing import Dict

from agents.cache import AgentCache
from agents.disaster_monitor_agent import DisasterMonitorAgent
from agents.image_scrape_agent import ImageScrapeAgent
from agents.image_validator_agent import ImageValidatorAgent
from agents.live_feed import publish_events
from config.agent_config import (
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_RUN_NAME,
    MLFLOW_TRACKING_URI,
    REACT_TRACE_PATH,
    TARGET_MIN,
)
from step1_collect_data import DISASTER_CLASSES, dataset_counts


class CoordinatorAgent:
    def __init__(self):
        self.cache = AgentCache()
        self.monitor_agent = DisasterMonitorAgent()
        self.scrape_agent = ImageScrapeAgent(self.cache)
        self.validator_agent = ImageValidatorAgent()
        self.events_detected_total = 0
        self.events_processed_total = 0
        self.images_collected_total = 0
        self.agent_collected: Dict[str, int] = {cls_name: 0 for cls_name in DISASTER_CLASSES}
        os.makedirs(os.path.dirname(REACT_TRACE_PATH), exist_ok=True)

    def run(self) -> None:
        try:
            self.log_thought("Checking current class distribution in raw_dataset/")
            counts = dataset_counts()
            under_target = {
                cls_name: TARGET_MIN - count
                for cls_name, count in counts.items()
                if count < TARGET_MIN and cls_name != "not_disaster"
            }
            self.log_thought(f"Under-target classes: {under_target}")

            if not under_target:
                self.log_observation("No disaster classes are under target; skipping live API polling for this run")
                self.log_thought("Run complete. Logging summary to MLflow.")
                self.log_to_mlflow(under_target)
                return

            self.log_thought("Fetching live disaster events from monitoring sources")
            events = self.monitor_agent.fetch_all()
            self.events_detected_total = len(events)
            publish_events(events)
            self.log_observation(f"Found {len(events)} candidate events across all sources")

            for event in events:
                if event.disaster_class not in under_target:
                    self.log_thought(f"Skipping {event.event_id}: class '{event.disaster_class}' already at target")
                    continue

                allow_zero_retry = under_target[event.disaster_class] > 0
                if self.cache.already_processed(event.event_id, allow_zero_retry=allow_zero_retry):
                    self.log_thought(f"Skipping {event.event_id}: already processed")
                    continue

                self.log_thought(
                    f"Event {event.event_id} ({event.disaster_class}) at "
                    f"{event.place_name or event.latitude}. Need {under_target[event.disaster_class]} more images. "
                    "Dispatching scrape agent."
                )
                self.log_action(f"search_twitter(event={event.event_id})")
                candidates = self.scrape_agent.collect_for_event(event)
                self.log_observation(f"{len(candidates)} raw image candidates found")

                self.log_action(f"validate_and_save(candidates={len(candidates)})")
                saved = self.validator_agent.process(candidates)
                self.log_observation(f"{saved} images passed validation and were saved")

                self.events_processed_total += 1
                self.images_collected_total += saved
                self.agent_collected[event.disaster_class] += saved
                under_target[event.disaster_class] -= saved
                self.cache.mark_processed(event.event_id, event.disaster_class, saved)

                if under_target[event.disaster_class] <= 0:
                    self.log_thought(
                        f"Class '{event.disaster_class}' now at target, no further action needed for this class"
                    )

            self.log_thought("Run complete. Logging summary to MLflow.")
            self.log_to_mlflow(under_target)
        finally:
            self.cache.close()

    def log_thought(self, message: str) -> None:
        self._log("THOUGHT", message)

    def log_action(self, message: str) -> None:
        self._log("ACTION", message)

    def log_observation(self, message: str) -> None:
        self._log("OBSERVATION", message)

    def _log(self, prefix: str, message: str) -> None:
        line = f"[{prefix}] {message}"
        print(line)
        with open(REACT_TRACE_PATH, "a", encoding="utf-8") as fobj:
            fobj.write(line + "\n")

    def log_to_mlflow(self, under_target: Dict[str, int]) -> None:
        import mlflow

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        with mlflow.start_run(run_name=MLFLOW_RUN_NAME):
            mlflow.log_metric("events_detected_total", self.events_detected_total)
            mlflow.log_metric("events_processed_total", self.events_processed_total)
            mlflow.log_metric("images_collected_total", self.images_collected_total)
            for cls_name in DISASTER_CLASSES:
                mlflow.log_metric(f"agent_collected_{cls_name}", self.agent_collected.get(cls_name, 0))
                if cls_name in under_target:
                    mlflow.log_metric(f"agent_remaining_needed_{cls_name}", max(0, under_target[cls_name]))
            if os.path.exists(REACT_TRACE_PATH):
                try:
                    mlflow.log_artifact(REACT_TRACE_PATH)
                except OSError as exc:
                    self.log_observation(f"MLflow artifact upload skipped: {exc}")
                    mlflow.log_param("step0_react_trace_path", REACT_TRACE_PATH)
