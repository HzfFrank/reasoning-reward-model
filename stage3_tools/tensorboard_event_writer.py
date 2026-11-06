#!/usr/bin/env python3
"""Small TensorBoard scalar writer with no tensorboard package dependency.

The training environment currently lacks the ``tensorboard`` Python package, so
``torch.utils.tensorboard.SummaryWriter`` cannot be imported.  This module first
tries the normal SummaryWriter path and then falls back to writing TensorBoard's
simple scalar event format directly as TFRecord records.
"""

from __future__ import annotations

import json
import math
import socket
import struct
import time
from pathlib import Path
from typing import Any

try:
    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
except Exception:  # pragma: no cover - handled at runtime
    descriptor_pb2 = None
    descriptor_pool = None
    message_factory = None

_CRC32C_TABLE: list[int] | None = None


def _crc32c_table() -> list[int]:
    global _CRC32C_TABLE
    if _CRC32C_TABLE is not None:
        return _CRC32C_TABLE
    table = []
    poly = 0x82F63B78
    for i in range(256):
        crc = i
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ poly
            else:
                crc >>= 1
        table.append(crc & 0xFFFFFFFF)
    _CRC32C_TABLE = table
    return table


def _crc32c(data: bytes) -> int:
    crc = 0xFFFFFFFF
    table = _crc32c_table()
    for b in data:
        crc = table[(crc ^ b) & 0xFF] ^ (crc >> 8)
    return (~crc) & 0xFFFFFFFF


def _masked_crc32c(data: bytes) -> int:
    crc = _crc32c(data)
    return (((crc >> 15) | ((crc << 17) & 0xFFFFFFFF)) + 0xA282EAD8) & 0xFFFFFFFF


def _message_class(pool, name: str):
    desc = pool.FindMessageTypeByName(name)
    if hasattr(message_factory, "GetMessageClass"):
        return message_factory.GetMessageClass(desc)
    return message_factory.MessageFactory(pool).GetPrototype(desc)


def _build_event_classes():
    if descriptor_pb2 is None or descriptor_pool is None or message_factory is None:
        raise RuntimeError("google.protobuf is unavailable")

    file_proto = descriptor_pb2.FileDescriptorProto()
    file_proto.name = "minimal_tensorboard_event.proto"
    file_proto.package = "minimal_tb"
    file_proto.syntax = "proto3"

    summary = file_proto.message_type.add()
    summary.name = "Summary"
    value = summary.nested_type.add()
    value.name = "Value"

    field = value.field.add()
    field.name = "tag"
    field.number = 1
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_STRING

    field = value.field.add()
    field.name = "simple_value"
    field.number = 2
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT

    field = summary.field.add()
    field.name = "value"
    field.number = 1
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
    field.type_name = ".minimal_tb.Summary.Value"

    event = file_proto.message_type.add()
    event.name = "Event"

    field = event.field.add()
    field.name = "wall_time"
    field.number = 1
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE

    field = event.field.add()
    field.name = "step"
    field.number = 2
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_INT64

    field = event.field.add()
    field.name = "file_version"
    field.number = 3
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_STRING

    field = event.field.add()
    field.name = "summary"
    field.number = 5
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
    field.type_name = ".minimal_tb.Summary"

    pool = descriptor_pool.DescriptorPool()
    pool.Add(file_proto)
    return _message_class(pool, "minimal_tb.Event"), _message_class(
        pool, "minimal_tb.Summary"
    )


class MinimalEventFileWriter:
    """Write scalar TensorBoard events directly as TFRecord records."""

    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        stamp = int(time.time())
        host = socket.gethostname().split(".")[0]
        self.path = self.log_dir / f"events.out.tfevents.{stamp}.{host}.sole-r1"
        self._event_cls, self._summary_cls = _build_event_classes()
        self._fh = self.path.open("ab")
        event = self._event_cls()
        event.wall_time = time.time()
        event.file_version = "brain.Event:2"
        self._write_event(event)
        self._fh.flush()

    def _write_record(self, payload: bytes) -> None:
        length = struct.pack("<Q", len(payload))
        self._fh.write(length)
        self._fh.write(struct.pack("<I", _masked_crc32c(length)))
        self._fh.write(payload)
        self._fh.write(struct.pack("<I", _masked_crc32c(payload)))

    def _write_event(self, event) -> None:
        self._write_record(event.SerializeToString())

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        event = self._event_cls()
        event.wall_time = time.time()
        event.step = int(step)
        summary = self._summary_cls()
        item = summary.value.add()
        item.tag = str(tag)
        item.simple_value = float(value)
        event.summary.CopyFrom(summary)
        self._write_event(event)

    def flush(self) -> None:
        self._fh.flush()

    def close(self) -> None:
        self._fh.flush()
        self._fh.close()


class TensorBoardMetricWriter:
    """JSONL + TensorBoard scalar writer compatible with v1/v2 train loops."""

    def __init__(self, run_dir: Path, enabled: bool):
        self.run_dir = Path(run_dir)
        self.jsonl = self.run_dir / "metrics.jsonl"
        self.tb = None
        self.tb_error = None
        self.tensorboard_dir = self.run_dir / "tensorboard"
        if enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.tb = SummaryWriter(log_dir=str(self.tensorboard_dir))
            except Exception as torch_exc:
                try:
                    self.tb = MinimalEventFileWriter(self.tensorboard_dir)
                    self.tb_error = f"torch SummaryWriter unavailable; using built-in event writer ({type(torch_exc).__name__}: {torch_exc})"
                except Exception as fallback_exc:
                    self.tb_error = f"TensorBoard unavailable: {type(torch_exc).__name__}: {torch_exc}; fallback failed: {type(fallback_exc).__name__}: {fallback_exc}"
                    self.tb = None

    def write(self, step: int, metrics: dict[str, Any]) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with self.jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"step": step, **metrics}, ensure_ascii=False) + "\n")
        if self.tb is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    self.tb.add_scalar(str(key), float(value), step)
            self.tb.flush()

    def close(self) -> None:
        if self.tb is not None:
            self.tb.close()
