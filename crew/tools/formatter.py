from crew.report.facts import FactsBlock
from crew.report.template import render_report

def format_markdown(facts: FactsBlock) -> str: return render_report(facts)
class MarkdownFormatterTool:
    def run(self, facts: FactsBlock) -> str: return render_report(facts)
