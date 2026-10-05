from crew.sources.reliefweb import ReliefWebSource

class ReliefWebEventSearchTool:
    name = "reliefweb_event_search"
    def run(self, location: str):
        events = ReliefWebSource().search(location)
        return events[0] if events else ReliefWebSource.fallback(location)
