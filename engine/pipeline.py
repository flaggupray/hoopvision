from dataclasses import dataclass, field
import time
from engine.decoder import VideoDecoder, DecoderConfig
from engine.detector import Detector, DetectorConfig
from engine.ocr import JerseyOCR, OCRConfig
from engine.classifier import ActionClassifier, ClassifierConfig
from engine.compiler import TimelineCompiler, CompilerConfig
from engine.schema import Player, GameMetadata, Timeline, BoundingBox


@dataclass
class PipelineConfig:
    sample_rate: int = 6
    device: str = "auto"
    enable_ocr: bool = True
    enable_classifier: bool = True
    confidence_threshold: float = 0.5
    max_events: int = 500
    dedup_window_seconds: float = 3.0
    ocr_config: OCRConfig = field(default_factory=OCRConfig)
    detector_config: DetectorConfig | None = None
    classifier_config: ClassifierConfig | None = None
    compiler_config: CompilerConfig | None = None


class Pipeline:
    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self._stats: dict = {}

    def run(self, video_path: str, metadata: GameMetadata | None = None) -> Timeline:
        start_time = time.time()
        metadata = metadata or GameMetadata()
        stats = {"total_frames": 0, "total_detections": 0, "total_events": 0}

        decoder_config = DecoderConfig(sample_rate=self.config.sample_rate)
        decoder = VideoDecoder(video_path, decoder_config)

        detector_config = self.config.detector_config or DetectorConfig(
            confidence_threshold=self.config.confidence_threshold,
            device=self.config.device,
        )
        detector = Detector(detector_config)

        ocr = JerseyOCR(self.config.ocr_config) if self.config.enable_ocr else None
        classifier = ActionClassifier(self.config.classifier_config) if self.config.enable_classifier else None
        compiler = TimelineCompiler(self.config.compiler_config)

        all_events = []
        players: dict[int, Player] = {}

        for frame, timestamp in decoder.iter_with_timestamps():
            stats["total_frames"] += 1

            detections = detector.detect(frame, stats["total_frames"], timestamp)
            stats["total_detections"] += len(detections)

            if self.config.enable_ocr and ocr:
                for d in detections:
                    if d.class_ == "person" and d.track_id not in players:
                        number = ocr.read_jersey_number(frame, d.bbox, d.track_id)
                        players[d.track_id] = Player(
                            track_id=d.track_id,
                            jersey_number=number,
                            bbox=d.bbox,
                        )

            if self.config.enable_classifier and classifier:
                player_list = list(players.values())
                events = classifier.classify_window(detections, player_list)
                for ev in events:
                    if ev.player_track_id in players:
                        ev.player_number = players[ev.player_track_id].jersey_number
                all_events.extend(events)
                stats["total_events"] += len(events)

        if not self.config.enable_classifier:
            all_events = classifier.get_history() if classifier else []

        decoder.close()

        timeline = compiler.compile(all_events, metadata)
        stats["elapsed_seconds"] = time.time() - start_time
        self._stats = stats
        return timeline

    def stats(self) -> dict:
        return dict(self._stats)
