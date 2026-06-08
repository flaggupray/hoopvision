from dataclasses import dataclass, field
import time
from engine.decoder import VideoDecoder, DecoderConfig
from engine.detector import Detector, DetectorConfig
from engine.ocr import JerseyOCR, OCRConfig
from engine.classifier import ActionClassifier, ClassifierConfig
from engine.compiler import TimelineCompiler, CompilerConfig
from engine.pose import PoseEstimator, PoseConfig
from engine.schema import Player, GameMetadata, Timeline


@dataclass
class PipelineConfig:
    sample_rate: int = 15
    device: str = "cpu"
    enable_ocr: bool = True
    enable_classifier: bool = True
    enable_pose: bool = True
    pose_every_n_frames: int = 3
    confidence_threshold: float = 0.5
    max_events: int = 500
    dedup_window_seconds: float = 3.0
    ocr_config: OCRConfig = field(default_factory=OCRConfig)
    detector_config: DetectorConfig | None = None
    classifier_config: ClassifierConfig | None = None
    compiler_config: CompilerConfig | None = None
    pose_config: PoseConfig | None = None


class Pipeline:
    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self._stats: dict = {}

    def run(self, video_path: str, metadata: GameMetadata | None = None) -> Timeline:
        t0 = time.time()
        metadata = metadata or GameMetadata()
        stats = {"total_frames": 0, "total_detections": 0, "total_events": 0,
                 "ball_detections": 0, "pose_frames": 0}

        decoder = VideoDecoder(video_path, DecoderConfig(sample_rate=self.config.sample_rate))
        detector = Detector(self.config.detector_config or DetectorConfig(
            confidence_threshold=self.config.confidence_threshold, device=self.config.device))
        ocr = JerseyOCR(self.config.ocr_config) if self.config.enable_ocr else None
        classifier = ActionClassifier(self.config.classifier_config) if self.config.enable_classifier else None
        compiler = TimelineCompiler(self.config.compiler_config)
        pose_estimator = PoseEstimator(self.config.pose_config) if self.config.enable_pose else None

        all_events: list = []
        players: dict[int, Player] = {}
        hoop_cache: dict = {}
        initialized = False

        for frame, timestamp in decoder.iter_with_timestamps():
            stats["total_frames"] += 1
            fnum = stats["total_frames"]

            if not initialized:
                h, w = frame.shape[:2]
                if classifier:
                    classifier.config.frame_width = w
                    classifier.config.frame_height = h
                hoop_cache = {
                    "cx": w * (classifier.config.hoop_x_ratio if classifier else 0.5),
                    "cy": h * (classifier.config.hoop_y_ratio if classifier else 0.22),
                    "r": classifier.config.hoop_zone_radius if classifier else 150,
                }
                initialized = True
                print(f"   {w}x{h} hoop@({hoop_cache['cx']:.0f},{hoop_cache['cy']:.0f})", flush=True)

            # Detection
            detections = detector.detect(frame, fnum, timestamp)
            stats["total_detections"] += len(detections)
            balls = [d for d in detections if d.class_ == "ball"]
            stats["ball_detections"] += len(balls)

            # OCR (only for new players, cached by track_id)
            if ocr:
                for d in detections:
                    if d.class_ == "person" and d.track_id not in players:
                        number = ocr.read_jersey_number(frame, d.bbox, d.track_id)
                        players[d.track_id] = Player(
                            track_id=d.track_id, jersey_number=number, bbox=d.bbox)

            # Pose: only run every N frames OR when players are near hoop
            pose_results: dict = {}
            persons = [d for d in detections if d.class_ == "person"]
            should_pose = self.config.enable_pose and pose_estimator and persons and (
                fnum % self.config.pose_every_n_frames == 0 or
                any(np.sqrt((p.bbox.center[0] - hoop_cache["cx"]) ** 2 +
                            (p.bbox.center[1] - hoop_cache["cy"]) ** 2) < hoop_cache["r"] * 1.5
                    for p in persons)
            )
            if should_pose:
                stats["pose_frames"] += 1
                person_bboxes = [(d.track_id, d.bbox) for d in persons]
                pose_results = pose_estimator.estimate(frame, person_bboxes)

            # Optical flow
            flow_balls = detector.ball_tracker.track(
                frame, (hoop_cache["cx"], hoop_cache["cy"]), hoop_cache["r"])

            # Classification
            if classifier:
                events = classifier.classify_frame(
                    detections, list(players.values()), pose_results, flow_balls)
                for ev in events:
                    if ev.player_track_id in players:
                        ev.player_number = players[ev.player_track_id].jersey_number
                all_events.extend(events)
                stats["total_events"] += len(events)

            if fnum % 15 == 0:
                print(f"   f{fnum}: {stats['total_detections']}det "
                      f"{stats['ball_detections']}ball {stats['total_events']}ev "
                      f"(pose:{stats['pose_frames']})", flush=True)

        decoder.close()
        print(f"   raw:{len(all_events)} compiled:{len(all_events)}", flush=True)
        timeline = compiler.compile(all_events, metadata)
        stats["elapsed_seconds"] = time.time() - t0
        self._stats = stats
        return timeline

    def stats(self) -> dict:
        return dict(self._stats)
