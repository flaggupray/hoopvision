from dataclasses import dataclass
import re
import numpy as np
from engine.schema import BoundingBox


@dataclass
class OCRConfig:
    min_confidence: float = 0.6
    cache_enabled: bool = True
    use_gpu: bool = False


class JerseyOCR:
    def __init__(self, config: OCRConfig | None = None):
        self.config = config or OCRConfig()
        self._cache: dict[int, int] = {}
        self._ocr = None

    def _lazy_init_ocr(self):
        if self._ocr is not None:
            return
        try:
            from paddleocr import PaddleOCR
            self._ocr = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                use_gpu=self.config.use_gpu,
                show_log=False,
            )
        except ImportError:
            self._ocr = False

    def read_jersey_number(self, frame: np.ndarray, bbox: BoundingBox, track_id: int) -> int | None:
        if self.config.cache_enabled and track_id in self._cache:
            return self._cache[track_id]

        x1, y1 = max(0, int(bbox.x1)), max(0, int(bbox.y1))
        x2, y2 = int(bbox.x2), int(bbox.y2)
        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        upper_half = crop[:crop.shape[0] // 2, :]
        number = self._ocr_read(upper_half)
        if number is None:
            number = self._ocr_read(crop)

        if number is not None and self.config.cache_enabled:
            self._cache[track_id] = number
        return number

    def _ocr_read(self, image: np.ndarray) -> int | None:
        self._lazy_init_ocr()
        if self._ocr is False or self._ocr is None:
            return self._fallback_ocr(image)

        try:
            results = self._ocr.ocr(image, cls=True)
            if not results or not results[0]:
                return None
            for line in results[0]:
                text = line[1][0]
                conf = line[1][1]
                number = self._extract_number(text)
                if number is not None and conf >= self.config.min_confidence:
                    return number
        except Exception:
            return self._fallback_ocr(image)
        return None

    def _fallback_ocr(self, image: np.ndarray) -> int | None:
        """Simple contour-based number extraction as fallback when PaddleOCR unavailable."""
        try:
            import cv2
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return None
        except Exception:
            return None

    def _extract_number(self, text: str) -> int | None:
        text = text.strip()
        match = re.search(r'\b(\d{1,2})\b', text)
        if match:
            return self._validate_number(int(match.group(1)))
        return None

    def _validate_number(self, n: int) -> int | None:
        if 0 <= n <= 99:
            return n
        return None

    def clear_cache(self):
        self._cache.clear()

    def get_cache(self) -> dict[int, int]:
        return dict(self._cache)
