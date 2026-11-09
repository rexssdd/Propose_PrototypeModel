"""Local, opt-in packet-to-flow capture service for the LCC dashboard.

This service is intentionally separate from Laravel. It binds to localhost,
requires an explicit interface, and emits the same nine-column schema used by
ml/predict.py. The live features are flow-time approximations; CSV replay
remains the benchmark path because captured traffic has no ground-truth label.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from scapy.all import AsyncSniffer, IP, IPv6, TCP, UDP  # type: ignore
except ImportError:  # keep /health usable when Npcap/Scapy is not installed
    AsyncSniffer = None  # type: ignore[assignment]
    IP = IPv6 = TCP = UDP = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[1]
ML_ROOT = ROOT / "app" / "ml"
import sys

sys.path.insert(0, str(ML_ROOT))
import predict  # noqa: E402


FEATURES = ["sttl", "ct_state_ttl", "dload", "rate", "dmean", "dttl", "ackdat", "sload", "sbytes"]
BUNDLE_PATH = Path(os.getenv("ML_BUNDLE", ROOT / "storage" / "app" / "ml" / "hybrid_if_svm_inference_bundle.joblib"))
IDLE_SECONDS = float(os.getenv("LIVE_FLOW_IDLE_SECONDS", "1.5"))
INFERENCE_SECONDS = float(os.getenv("LIVE_INFERENCE_SECONDS", "2"))
ALLOWED_ORIGINS = [x.strip() for x in os.getenv("LIVE_CORS_ORIGINS", "http://127.0.0.1:8000,http://localhost:8000").split(",") if x.strip()]


class StartRequest(BaseModel):
    interface: str = Field(default="", max_length=128)


@dataclass
class Flow:
    key: tuple[Any, ...]
    source: tuple[str, int]
    destination: tuple[str, int]
    protocol: int
    started: float
    ended: float
    src_packets: int = 0
    dst_packets: int = 0
    src_bytes: int = 0
    dst_bytes: int = 0
    src_ttls: list[int] = field(default_factory=list)
    dst_ttls: list[int] = field(default_factory=list)
    dst_sizes: list[int] = field(default_factory=list)
    syn_at: float | None = None
    ack_at: float | None = None
    emitted: bool = False

    def add(self, timestamp: float, source_side: bool, size: int, ttl: int, flags: str = "") -> None:
        self.ended = timestamp
        if source_side:
            self.src_packets += 1
            self.src_bytes += size
            self.src_ttls.append(ttl)
            if "S" in flags and "A" not in flags and self.syn_at is None:
                self.syn_at = timestamp
        else:
            self.dst_packets += 1
            self.dst_bytes += size
            self.dst_ttls.append(ttl)
            self.dst_sizes.append(size)
            if "A" in flags and self.ack_at is None:
                self.ack_at = timestamp

    def features(self) -> dict[str, float]:
        duration = max(self.ended - self.started, 0.001)
        total_packets = self.src_packets + self.dst_packets
        ackdat = self.ack_at - self.syn_at if self.syn_at is not None and self.ack_at is not None else 0.0
        return {
            "sttl": float(sum(self.src_ttls) / len(self.src_ttls)) if self.src_ttls else 0.0,
            "ct_state_ttl": float(1 if self.protocol == 6 and self.dst_packets else 0),
            "dload": float(self.dst_bytes / duration),
            "rate": float(total_packets / duration),
            "dmean": float(sum(self.dst_sizes) / len(self.dst_sizes)) if self.dst_sizes else 0.0,
            "dttl": float(sum(self.dst_ttls) / len(self.dst_ttls)) if self.dst_ttls else 0.0,
            "ackdat": float(max(ackdat, 0.0)),
            "sload": float(self.src_bytes / duration),
            "sbytes": float(self.src_bytes),
        }


class CaptureState:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.session_id: str | None = None
        self.interface = ""
        self.running = False
        self.started_at: float | None = None
        self.packets = 0
        self.flows: dict[tuple[Any, ...], Flow] = {}
        self.predictions: list[dict[str, Any]] = []
        self.last_error = ""
        self.sniffer: Any = None
        self.worker: threading.Thread | None = None

    def reset(self, interface: str) -> None:
        self.session_id = str(uuid.uuid4())
        self.interface = interface
        self.running = True
        self.started_at = time.time()
        self.packets = 0
        self.flows = {}
        self.predictions = []
        self.last_error = ""


state = CaptureState()
app = FastAPI(title="LCC live flow capture", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"])


def _packet_parts(packet: Any) -> tuple[str, str, int, int, int, int, str] | None:
    layer = IP if IP is not None and packet.haslayer(IP) else IPv6 if IPv6 is not None and packet.haslayer(IPv6) else None
    if layer is None:
        return None
    network = packet[layer]
    src, dst, ttl = str(network.src), str(network.dst), int(getattr(network, "ttl", getattr(network, "hlim", 0)))
    protocol = int(getattr(network, "proto", getattr(network, "nh", 0)))
    sport = dport = 0
    flags = ""
    if TCP is not None and packet.haslayer(TCP):
        transport = packet[TCP]
        sport, dport, protocol, flags = int(transport.sport), int(transport.dport), 6, str(transport.flags)
    elif UDP is not None and packet.haslayer(UDP):
        transport = packet[UDP]
        sport, dport, protocol = int(transport.sport), int(transport.dport), 17
    return src, dst, sport, dport, protocol, len(packet), flags


def on_packet(packet: Any) -> None:
    parts = _packet_parts(packet)
    if parts is None:
        return
    src, dst, sport, dport, protocol, size, flags = parts
    now = time.time()
    forward = (src, sport, dst, dport)
    reverse = (dst, dport, src, sport)
    key = (min(forward, reverse), max(forward, reverse), protocol)
    with state.lock:
        flow = state.flows.get(key)
        if flow is None:
            flow = Flow(key, (src, sport), (dst, dport), protocol, now, now)
            state.flows[key] = flow
        source_side = (src, sport) == flow.source
        flow.add(now, source_side, size, int(getattr(packet[IP if IP is not None and packet.haslayer(IP) else IPv6], "ttl", 0)), flags)
        state.packets += 1


def infer_completed() -> None:
    with state.lock:
        now = time.time()
        ready = [flow for flow in state.flows.values() if not flow.emitted and now - flow.ended >= IDLE_SECONDS]
        for flow in ready:
            flow.emitted = True
        session_id = state.session_id
    if not ready or not session_id:
        return
    try:
        if not BUNDLE_PATH.is_file():
            raise RuntimeError(f"Model bundle not found at {BUNDLE_PATH}")
        bundle = predict.joblib.load(BUNDLE_PATH)
        frame = pd.DataFrame([flow.features() for flow in ready], columns=FEATURES)
        result = predict.analyse(frame, bundle)
        live_rows = []
        for index, flow in enumerate(ready):
            row = result["preview"][index]
            row["live_flow"] = {"source": flow.source[0], "destination": flow.destination[0], "protocol": flow.protocol, "packets": flow.src_packets + flow.dst_packets}
            live_rows.append(row)
        with state.lock:
            state.predictions.extend(live_rows)
            state.predictions = state.predictions[-500:]
            state.last_error = ""
    except Exception as exc:  # keep capture alive and surface the exact issue
        with state.lock:
            state.last_error = str(exc)


def capture_loop() -> None:
    try:
        if AsyncSniffer is None:
            raise RuntimeError("Scapy is not installed. Install capture_api/requirements.txt and Npcap on Windows.")
        state.sniffer = AsyncSniffer(iface=state.interface or None, prn=on_packet, store=False)
        state.sniffer.start()
        while state.running:
            infer_completed()
            time.sleep(INFERENCE_SECONDS)
    except Exception as exc:
        with state.lock:
            state.last_error = str(exc)
            state.running = False
    finally:
        infer_completed()
        with state.lock:
            sniffer = state.sniffer
            state.sniffer = None
        if sniffer is not None:
            try:
                sniffer.stop()
            except Exception:
                pass


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "scapy_installed": AsyncSniffer is not None, "bundle_ready": BUNDLE_PATH.is_file(), "features": FEATURES}


@app.post("/capture/start")
def start_capture(request: StartRequest) -> dict[str, Any]:
    with state.lock:
        if state.running:
            raise HTTPException(status_code=409, detail="Capture is already running.")
        state.reset(request.interface.strip())
        state.worker = threading.Thread(target=capture_loop, daemon=True)
        state.worker.start()
        return status_payload()


@app.post("/capture/stop")
def stop_capture() -> dict[str, Any]:
    with state.lock:
        state.running = False
        worker = state.worker
    if worker and worker.is_alive():
        worker.join(timeout=5)
    return status_payload()


@app.get("/capture/status")
def capture_status() -> dict[str, Any]:
    infer_completed()
    return status_payload()


def status_payload() -> dict[str, Any]:
    with state.lock:
        return {"session_id": state.session_id, "running": state.running, "interface": state.interface, "started_at": state.started_at, "packets": state.packets, "flows": len(state.flows), "predictions": state.predictions[-100:], "last_error": state.last_error, "feature_mode": "live flow approximation; not equivalent to labelled UNSW-NB15 rows"}
