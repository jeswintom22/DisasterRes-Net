"""Optional CrewAI configuration. The deterministic runner is always available."""
from __future__ import annotations
from .tools import MetricAggregatorTool, MarkdownFormatterTool
from .tools.image_dl import ImageDownloaderTool
from .tools.image_search import GoogleImageScrapeTool
from .tools.reliefweb import ReliefWebEventSearchTool
from .tools.pipeline_tool import DisasterResPipelineTool, DamageAssessmentTool

def build_agents():
    try:
        from crewai import Agent
    except ImportError:
        return None
    return [
        Agent(role="Disaster Intelligence Scout", goal="Find authoritative event context and suitable imagery.", backstory="You are an emergency intelligence officer.", tools=[ReliefWebEventSearchTool(), GoogleImageScrapeTool(), ImageDownloaderTool()], allow_delegation=False),
        Agent(role="Damage Analyst", goal="Analyse only validated imagery.", backstory="You are a computer vision analyst.", tools=[DisasterResPipelineTool(), DamageAssessmentTool(), MetricAggregatorTool()], allow_delegation=False),
        Agent(role="Disaster Report Writer", goal="Communicate only supplied facts.", backstory="You write concise emergency situation reports.", tools=[MarkdownFormatterTool()], allow_delegation=False),
    ]
