from dataclasses import dataclass, field
import time
import sys
from engine.decoder import VideoDecoder, DecoderConfig
from engine.detector import Detector, DetectorConfig
from engine.ocr import JerseyOCR, OCRConfig
from engine.classifier import ActionClassifier, ClassifierConfig
from engine.compiler import TimelineCompiler, CompilerConfig
from engine.pose import PoseEstimator, PoseConfig
from engine.schema import Player, GameMetadata, Timeline, BoundingBox


@dataclass
class PipelineConfig:
    sample_rate: int = 6
    device: str = "cpu"
    enable_ocr: bool = True
    enable_classifier: bool = True
    enable_pose: bool = True
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
        start_time = time.time()
        metadata = metadata or GameMetadata()
        stats = {"total_frames": 0, "total_detections": 0, "total_events": 0,
                 "ball_detections": 0, "pose_detections": 0}

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
        pose_estimator = PoseEstimator(self.config.pose_config) if self.config.enable_pose else None

        all_events = []
        players: dict[int, Player] = {}
        frame_dimensions_set = False

        for frame, timestamp in decoder.iter_with_timestamps():
            stats["total_frames"] += 1

            if not frame_dimensions_set:
                h, w = frame.shape[:2]
                if classifier is not None:
                    classifier.config.frame_width = w
                    classifier.config.frame_height = h
                frame_dimensions_set = True
                print(f"   Frame size: {w}x{h}", flush=True)

            # Detection
            detections = detector.detect(frame, stats["total_frames"], timestamp)
            stats["total_detections"] += len(detections)
            balls = [d for d in detections if d.class_ == "ball"]
            stats["ball_detections"] += len(balls)

            # OCR
            if self.config.enable_ocr and ocr:
                for d in detections:
                    if d.class_ == "person" and d.track_id not in players:
                        number = ocr.read_jersey_number(frame, d.bbox, d.track_id)
                        players[d.track_id] = Player(
                            track_id=d.track_id, jersey_number=number, bbox=d.bbox)

            # Pose estimation
            pose_results: dict[int, any] = {}
            if self.config.enable_pose and pose_estimator:
                person_bboxes = [(d.track_id, d.bbox) for d in detections if d.class_ == "person"]
                if person_bboxes:
                    pose_results = pose_estimator.estimate(frame, person_bboxes)
                    shooting_count = sum(1 for pr in pose_results.values() if pr.shooting_pose)
                    stats["pose_detections"] += shooting_count

            # Optical flow ball tracking
            hoop_cx = frame.shape[1] * (classifier.config.hoop_x_ratio if classifier else 0.5)
            hoop_cy = frame.shape[0] * (classifier.config.hoop_y_ratio if classifier else 0.22)
            hoop_r = classifier.config.hoop_zone_radius if classifier else 150
            flow_balls = detector.ball_tracker.track(frame, (hoop_cx, hoop_cy), hoop_r)

            # Classification
            if self.config.enable_classifier and classifier:
                player_list = list(players.values())
                events = classifier.classify_frame(detections, player_list, pose_results, flow_balls)
                for ev in events:
                    if ev.player_track_id in players:
                        ev.player_number = players[ev.player_track_id].jersey_number
                all_events.extend(events)
                stats["total_events"] += len(events)

            if stats["total_frames"] % 30 == 0:
                print(f"   Progress: {stats['total_frames']} frames, "
                      f"{stats['total_detections']} det, "
                      f"{stats['ball_detections']} balls, "
                      f"{stats['total_events']} events", flush=True)

        decoder.close()

        print(f"   Raw events: {len(all_events)}", flush=True)
        timeline = compiler.compile(all_events, metadata)
        print(f"   Compiled: {len(timeline.events)} events", flush=True)
        stats["elapsed_seconds"] = time.time() - start_time
        self._stats = stats
        return timeline

    def stats(self) -> dict:
        return dict(self._stats)
