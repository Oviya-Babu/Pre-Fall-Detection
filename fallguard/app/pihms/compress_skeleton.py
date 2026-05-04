"""
Skeleton Data Compression — Delta encoding + gzip for skeleton keypoint data.
Targets 60% reduction in storage vs. raw float32 frames.
"""

import numpy as np
import gzip
import struct
import logging
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)

POSITION_SCALE = 32000
CONFIDENCE_SCALE = 255
MAGIC_KEYFRAME = b'\x4b\x46'
MAGIC_DELTA = b'\x44\x46'
MAGIC_BATCH = b'\x50\x42'
NUM_KEYPOINTS = 17
NUM_VALUES = 3


def quantize_frame(kp: np.ndarray) -> np.ndarray:
    q = np.zeros((NUM_KEYPOINTS, NUM_VALUES), dtype=np.int16)
    q[:, 0] = np.clip(kp[:, 0] * POSITION_SCALE, -32768, 32767).astype(np.int16)
    q[:, 1] = np.clip(kp[:, 1] * POSITION_SCALE, -32768, 32767).astype(np.int16)
    q[:, 2] = np.clip(kp[:, 2] * CONFIDENCE_SCALE, 0, 255).astype(np.int16)
    return q


def dequantize_frame(q: np.ndarray) -> np.ndarray:
    kp = np.zeros((NUM_KEYPOINTS, NUM_VALUES), dtype=np.float32)
    kp[:, 0] = q[:, 0].astype(np.float32) / POSITION_SCALE
    kp[:, 1] = q[:, 1].astype(np.float32) / POSITION_SCALE
    kp[:, 2] = q[:, 2].astype(np.float32) / CONFIDENCE_SCALE
    return kp


def encode_keyframe(kp: np.ndarray) -> bytes:
    return MAGIC_KEYFRAME + quantize_frame(kp).tobytes()


def decode_keyframe(data: bytes) -> np.ndarray:
    q = np.frombuffer(data[2:], dtype=np.int16).reshape(NUM_KEYPOINTS, NUM_VALUES)
    return dequantize_frame(q)


def encode_delta(cur: np.ndarray, prev: np.ndarray) -> bytes:
    delta = (quantize_frame(cur) - quantize_frame(prev)).astype(np.int16)
    return MAGIC_DELTA + delta.tobytes()


def decode_delta(data: bytes, prev_q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    delta = np.frombuffer(data[2:], dtype=np.int16).reshape(NUM_KEYPOINTS, NUM_VALUES)
    cur_q = (prev_q + delta).astype(np.int16)
    return dequantize_frame(cur_q), cur_q.copy()


class SkeletonCompressor:
    """Streaming compressor: accumulates frames, emits gzip batches."""

    def __init__(self, keyframe_interval: int = 120, gzip_level: int = 6):
        self.keyframe_interval = keyframe_interval
        self.gzip_level = gzip_level
        self._buffer: List[bytes] = []
        self._frame_count = 0
        self._prev_q: Optional[np.ndarray] = None
        self._batch_count = 0

    def add_frame(self, kp: np.ndarray) -> Optional[bytes]:
        if self._batch_count == 0:
            enc = encode_keyframe(kp)
            self._prev_q = quantize_frame(kp)
        else:
            cur_q = quantize_frame(kp)
            delta = (cur_q - self._prev_q).astype(np.int16)
            enc = MAGIC_DELTA + delta.tobytes()
            self._prev_q = cur_q

        self._buffer.append(enc)
        self._frame_count += 1
        self._batch_count += 1

        if self._batch_count >= self.keyframe_interval:
            result = self._compress_batch()
            self._reset()
            return result
        return None

    def flush(self) -> Optional[bytes]:
        if not self._buffer:
            return None
        result = self._compress_batch()
        self._reset()
        return result

    def _compress_batch(self) -> bytes:
        n = len(self._buffer)
        header = MAGIC_BATCH + struct.pack('<II', n, self.keyframe_interval)
        parts = [header]
        for f in self._buffer:
            parts.append(struct.pack('<H', len(f)) + f)
        raw = b''.join(parts)
        compressed = gzip.compress(raw, compresslevel=self.gzip_level)
        logger.debug(f"Batch: {n} frames, {len(raw)}→{len(compressed)} bytes")
        return compressed

    def _reset(self):
        self._buffer = []
        self._batch_count = 0

    @property
    def total_frames(self) -> int:
        return self._frame_count


class SkeletonDecompressor:
    """Decompress a batch back to individual frames."""

    @staticmethod
    def decompress_batch(data: bytes) -> List[np.ndarray]:
        raw = gzip.decompress(data)
        assert raw[:2] == MAGIC_BATCH
        n, _ = struct.unpack('<II', raw[2:10])
        offset = 10
        frames = []
        prev_q = None
        for _ in range(n):
            flen = struct.unpack('<H', raw[offset:offset+2])[0]
            offset += 2
            fdata = raw[offset:offset+flen]
            offset += flen
            if fdata[:2] == MAGIC_KEYFRAME:
                kp = decode_keyframe(fdata)
                prev_q = quantize_frame(kp)
                frames.append(kp)
            elif fdata[:2] == MAGIC_DELTA:
                kp, prev_q = decode_delta(fdata, prev_q)
                frames.append(kp)
        return frames
