"""Run ledger for the Lambda instance: JSON (machine) + markdown (human).

Files: .lambda-run/mnist-ddpm-rf.json and .lambda-run/mnist-ddpm-rf.md
The ledger is only ever rewritten in place, never deleted.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

SLUG = "mnist-ddpm-rf"
DEFAULT_DIR = Path(__file__).resolve().parent.parent / ".lambda-run"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def cost_so_far(rate_per_hour: float, launched_at: str,
                end: str | None = None, now: datetime | None = None) -> tuple[float, float]:
    """Return (hours, dollars) between launch and end (or now)."""
    start = _parse(launched_at)
    stop = _parse(end) if end else (now or datetime.now(timezone.utc))
    hours = max((stop - start).total_seconds(), 0.0) / 3600.0
    return hours, hours * rate_per_hour


class Ledger:
    def __init__(self, directory: Path | str = DEFAULT_DIR, slug: str = SLUG):
        self.dir = Path(directory)
        self.json_path = self.dir / f"{slug}.json"
        self.md_path = self.dir / f"{slug}.md"

    def read(self) -> dict:
        try:
            return json.loads(self.json_path.read_text())
        except FileNotFoundError:
            return {"instances": {}}

    def _write(self, data: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.json_path.write_text(json.dumps(data, indent=2) + "\n")
        self.md_path.write_text(self._render(data))

    def _render(self, data: dict) -> str:
        lines = [f"# Lambda run ledger ({SLUG})", ""]
        for iid, r in data["instances"].items():
            lines += [
                f"## Instance `{iid}`",
                f"- type: {r.get('type')}  region: {r.get('region')}  ${r.get('rate_per_hour')}/h",
                f"- launched: {r.get('launched_at')}  ip: {r.get('ip')}",
                f"- terminated: {r.get('terminated_at')}  final cost: {r.get('final_cost')}",
                "- events:",
            ]
            lines += [f"  - {e['time']} {e['event']}" for e in r.get("events", [])]
            lines.append("")
        return "\n".join(lines)

    def _update(self, iid: str, **fields) -> dict:
        data = self.read()
        rec = dict(data["instances"].get(iid, {"events": []}))
        event = fields.pop("event", None)
        rec.update(fields)
        if event:
            rec["events"] = [*rec.get("events", []), {"time": now_iso(), "event": event}]
        data["instances"] = {**data["instances"], iid: rec}
        self._write(data)
        return rec

    def record_launch(self, iid: str, itype: str, region: str, rate: float) -> dict:
        return self._update(iid, type=itype, region=region, rate_per_hour=rate,
                            launched_at=now_iso(), ip=None, terminated_at=None,
                            final_cost=None, event="launched")

    def record_status(self, iid: str, status: str, ip: str | None = None) -> dict:
        fields = {"event": f"status={status}"}
        if ip:
            fields["ip"] = ip
        return self._update(iid, **fields)

    def record_terminate(self, iid: str) -> dict:
        rec = self.read()["instances"].get(iid)
        fields: dict = {"terminated_at": now_iso(), "event": "terminate requested"}
        if rec and rec.get("launched_at"):
            _, dollars = cost_so_far(rec["rate_per_hour"], rec["launched_at"], fields["terminated_at"])
            fields["final_cost"] = round(dollars, 4)
        return self._update(iid, **fields)

    def cost(self, iid: str, now: datetime | None = None) -> tuple[float, float]:
        rec = self.read()["instances"].get(iid)
        if not rec or not rec.get("launched_at"):
            raise KeyError(f"instance {iid} not in ledger")
        return cost_so_far(rec["rate_per_hour"], rec["launched_at"],
                           rec.get("terminated_at"), now)
