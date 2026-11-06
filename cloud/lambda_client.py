#!/usr/bin/env python3
"""Stdlib-only Lambda Cloud CLI. Token is read from $LAMBDA_API_KEY and never printed."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ledger import Ledger  # noqa: E402

BASE = "https://cloud.lambda.ai/api/v1"
USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


class ApiError(RuntimeError):
    pass


def build_request(method: str, path: str, token: str, body: dict | None = None) -> urllib.request.Request:
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Authorization": f"Bearer {token}", "User-Agent": USER_AGENT,
               "Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    return urllib.request.Request(BASE + path, data=data, headers=headers, method=method)


def api(method: str, path: str, body: dict | None = None) -> dict:
    token = os.environ.get("LAMBDA_API_KEY")
    if not token:
        raise ApiError("LAMBDA_API_KEY is not set in the environment")
    req = build_request(method, path, token, body)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace").replace(token, "***")
        raise ApiError(f"HTTP {e.code} on {method} {path}: {detail}") from e
    except urllib.error.URLError as e:
        raise ApiError(f"network error on {method} {path}: {e.reason}") from e


def cmd_types(a) -> None:
    for name, v in sorted(api("GET", "/instance-types")["data"].items()):
        if a.filter and a.filter.lower() not in name.lower():
            continue
        t = v["instance_type"]
        regions = ",".join(r["name"] for r in v.get("regions_with_capacity_available", [])) or "none"
        print(f"{name}  ${t['price_cents_per_hour'] / 100:.2f}/h  regions={regions}  {t['description']}")


def cmd_instances(a) -> None:
    data = api("GET", "/instances")["data"]
    for i in data:
        price = i.get("instance_type", {}).get("price_cents_per_hour", 0) / 100
        print(f"{i['id']}  {i.get('instance_type', {}).get('name')}  {i['status']}  "
              f"ip={i.get('ip')}  ${price:.2f}/h  region={i.get('region', {}).get('name')}")
    if not data:
        print("(no instances)")


def cmd_ssh_keys(a) -> None:
    for k in api("GET", "/ssh-keys")["data"]:
        print(f"{k['id']}  {k['name']}")


def _rate_for(type_name: str) -> float:
    entry = api("GET", "/instance-types")["data"].get(type_name)
    if not entry:
        raise ApiError(f"unknown instance type {type_name}")
    return entry["instance_type"]["price_cents_per_hour"] / 100


def cmd_launch(a) -> None:
    rate = _rate_for(a.type)
    body = {"region_name": a.region, "instance_type_name": a.type,
            "ssh_key_names": [a.ssh_key], "quantity": 1, "name": a.name}
    iid = api("POST", "/instance-operations/launch", body)["data"]["instance_ids"][0]
    Ledger().record_launch(iid, a.type, a.region, rate)
    print(iid)


def cmd_wait(a) -> None:
    deadline = time.time() + a.timeout
    last = None
    while time.time() < deadline:
        inst = api("GET", f"/instances/{a.id}")["data"]
        status, ip = inst["status"], inst.get("ip")
        if status != last:
            Ledger().record_status(a.id, status, ip)
            last = status
        if status == "active" and ip:
            Ledger().record_status(a.id, "ready", ip)
            print(ip)
            return
        if status in ("terminated", "unhealthy"):
            raise ApiError(f"instance {a.id} entered status {status}")
        time.sleep(15)
    raise ApiError(f"timeout after {a.timeout}s waiting for {a.id}")


def cmd_terminate(a) -> None:
    if not a.yes:
        raise ApiError("refusing to terminate without --yes")
    api("POST", "/instance-operations/terminate", {"instance_ids": [a.id]})
    rec = Ledger().record_terminate(a.id)
    print(f"terminate requested for {a.id}; final cost ${rec.get('final_cost')}")


def cmd_cost(a) -> None:
    hours, dollars = Ledger().cost(a.id)
    print(f"hours={hours:.3f} dollars=${dollars:.2f}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Lambda Cloud helper (token from $LAMBDA_API_KEY)")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("types"); s.add_argument("--filter"); s.set_defaults(fn=cmd_types)
    sub.add_parser("instances").set_defaults(fn=cmd_instances)
    sub.add_parser("ssh-keys").set_defaults(fn=cmd_ssh_keys)
    s = sub.add_parser("launch")
    s.add_argument("--type", required=True); s.add_argument("--region", required=True)
    s.add_argument("--ssh-key", required=True); s.add_argument("--name", required=True)
    s.set_defaults(fn=cmd_launch)
    s = sub.add_parser("wait"); s.add_argument("--id", required=True)
    s.add_argument("--timeout", type=int, default=900); s.set_defaults(fn=cmd_wait)
    s = sub.add_parser("terminate"); s.add_argument("--id", required=True)
    s.add_argument("--yes", action="store_true"); s.set_defaults(fn=cmd_terminate)
    s = sub.add_parser("cost"); s.add_argument("--id", required=True); s.set_defaults(fn=cmd_cost)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.fn(args)
    except (ApiError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
