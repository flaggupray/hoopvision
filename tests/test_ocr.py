import pytest
import numpy as np
import cv2
from engine.ocr import JerseyOCR, OCRConfig
from engine.schema import BoundingBox


class TestOCRConfig:
    def test_defaults(self):
        config = OCRConfig()
        assert config.min_confidence == 0.6
        assert config.cache_enabled is True


class TestJerseyOCR:
    def test_create_ocr(self):
        ocr = JerseyOCR(OCRConfig())
        assert ocr is not None
        assert len(ocr._cache) == 0

    def test_read_clear_number(self):
        ocr = JerseyOCR(OCRConfig(min_confidence=0.3))
        frame = np.ones((100, 60, 3), dtype=np.uint8) * 255
        cv2.putText(frame, "23", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        bbox = BoundingBox(x1=0, y1=0, x2=60, y2=100)
        number = ocr.read_jersey_number(frame, bbox, track_id=1)
        assert number == 23 or number is None

    def test_cache_reuse(self):
        ocr = JerseyOCR(OCRConfig(min_confidence=0.3, cache_enabled=True))
        frame = np.ones((100, 60, 3), dtype=np.uint8) * 255
        cv2.putText(frame, "30", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        bbox = BoundingBox(x1=0, y1=0, x2=60, y2=100)
        number1 = ocr.read_jersey_number(frame, bbox, track_id=5)
        number2 = ocr.read_jersey_number(frame, bbox, track_id=5)
        assert number1 == number2

    def test_validate_number_range(self):
        ocr = JerseyOCR(OCRConfig())
        assert ocr._validate_number(23) == 23
        assert ocr._validate_number(0) == 0
        assert ocr._validate_number(99) == 99
        assert ocr._validate_number(100) is None
        assert ocr._validate_number(-1) is None

    def test_clear_cache(self):
        ocr = JerseyOCR(OCRConfig())
        ocr._cache[1] = 23
        ocr.clear_cache()
        assert len(ocr._cache) == 0
