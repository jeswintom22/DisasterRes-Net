from __future__ import annotations

def build_tasks(agents):
    try:
        from crewai import Task
    except ImportError:
        return []
    scout, analyst, narrator = agents
    task1 = Task(description="Find event context for {location}; never invent source facts.", expected_output="Structured source facts.", agent=scout)
    task2 = Task(description="Analyse only validated images from the scout context.", expected_output="Structured damage metrics.", agent=analyst, context=[task1])
    task3 = Task(description="Write a report using supplied facts only; do not state casualties.", expected_output="Markdown report.", agent=narrator, context=[task1, task2])
    return [task1, task2, task3]
