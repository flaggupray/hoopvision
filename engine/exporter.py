from dataclasses import dataclass
import json
import os
from engine.schema import Timeline
from engine.narrator import Narrator


@dataclass
class ExportConfig:
    template_dir: str = ""
    pretty_json: bool = True


class TimelineExporter:
    def __init__(self, config: ExportConfig | None = None):
        self.config = config or ExportConfig()
        self.narrator = Narrator()

    def export_html(self, timeline: Timeline) -> str:
        from jinja2 import Environment, FileSystemLoader, select_autoescape

        for event in timeline.events:
            if not hasattr(event, 'narrative') or event.narrative is None:
                event.narrative = self.narrator.narrate(event)

        template_dir = self.config.template_dir or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "shared", "templates"
        )

        env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(['html']),
        )
        template = env.get_template("timeline.html")
        return template.render(
            metadata=timeline.metadata,
            events=timeline.events,
        )

    def export_html_file(self, timeline: Timeline, output_path: str):
        html = self.export_html(timeline)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

    def export_json(self, timeline: Timeline) -> str:
        indent = 2 if self.config.pretty_json else None
        return json.dumps(
            timeline.model_dump(mode='json'),
            ensure_ascii=False,
            indent=indent,
        )
