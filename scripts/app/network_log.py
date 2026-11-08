#!/usr/bin/env python
"""
A network log of the board while Bias Lens works: evidence that no article text leaves the board.

    python scripts/app/network_log.py start --seconds 120   # start a capture on the board
    (meanwhile, analyse articles on the board's page)
    python scripts/app/network_log.py fetch                 # wait for it to end, then summarise it

The capture (sniff_network.py) runs in App Lab's Python image with the host's network and the
raw-socket capability, so it sees every interface: Wi-Fi, loopback and Docker's bridges. It keeps
addresses, ports and packet counts, never contents. The summary puts every flow in one group:

  app's web server     the app's container on port 7000: requests and answers
  app and the board    the app's container and other services on the board, such as the LED matrix's Bridge
  app and elsewhere    anything else the app's container sent or received. This should be empty.
  board, home network  the board's own traffic with the local network: discovery, DHCP and the like
  board, internet      the board's own traffic beyond the local network, listed in full
  inside the board     loopback and Docker's bridges, between the board's own services

It also counts the analyses that the app logged during the capture (counts only; the app never
logs text).

Results go to outputs/app/network_log/.
"""

import argparse
import ipaddress
import json
import os
import sys
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

from deploy_app import APP, IMAGE, PORT, Board, find_adb  # noqa: E402

REMOTE = "/home/arduino/.cache/bias-lens/network-log"
CAPTURE = "bias-lens-network-log"
APP_CONTAINER = f"{APP}-main-1"
INTERNAL = ("lo", "docker", "br-", "veth")  # interfaces that never leave the board


def is_local(address: str) -> bool:
    ip = ipaddress.ip_address(address)
    return ip.is_private or ip.is_link_local or ip.is_multicast or ip.is_loopback or address == "255.255.255.255"


def group(flow: Dict, app_ip: str, board_ips: List[str]) -> str:
    """Which of the groups in this file's description a flow belongs to."""
    src, dst = flow["src"], flow["dst"]
    if app_ip in (src, dst):
        app_port = flow["sport"] if src == app_ip else flow["dport"]
        other = dst if src == app_ip else src
        if app_port == PORT:
            return "app's web server"
        if other in board_ips:
            return "app and the board"
        return "app and elsewhere"
    if flow["interface"].startswith(INTERNAL):
        return "inside the board"
    remote = dst if flow["direction"] == "out" else src
    return "board, home network" if is_local(remote) else "board, internet"


def summarise(flows: List[Dict], app_ip: str, board_ips: List[str]) -> Dict:
    groups = {}
    for flow in flows:
        g = groups.setdefault(group(flow, app_ip, board_ips), {"flows": 0, "packets": 0, "bytes": 0})
        g["flows"] += 1
        g["packets"] += flow["packets"]
        g["bytes"] += flow["bytes"]
    listed = lambda name: sorted((f for f in flows if group(f, app_ip, board_ips) == name),  # noqa: E731
                                 key=lambda f: -f["packets"])
    return {"groups": groups, "app_and_elsewhere": listed("app and elsewhere"),
            "board_internet": listed("board, internet")}


def board_text(board: Board, command: str) -> str:
    return board.shell(command, capture=True).stdout.strip()


def start(board: Board, seconds: int, out: str) -> None:
    networks = "{{range .NetworkSettings.Networks}}{{.IPAddress}} {{.Gateway}}{{end}}"
    app_ip, gateway = board_text(board, f"docker inspect -f '{networks}' {APP_CONTAINER}").split()
    board_ips = board_text(board, "hostname -I").split() + [gateway]
    with open(os.path.join(out, "context.json"), "w", encoding="utf-8") as f:
        json.dump({"app_ip": app_ip, "board_ips": board_ips, "seconds": seconds}, f, indent=2)
    board.shell(f"docker rm -f {CAPTURE} >/dev/null 2>&1; rm -rf {REMOTE} && mkdir -p {REMOTE}")
    board.run("push", os.path.join(HERE, "sniff_network.py"), f"{REMOTE}/sniff_network.py")
    # Raw sockets need root inside the container (App Lab's image runs as another user by default);
    # the result is handed back to the board's user afterwards.
    board.shell(f"uid=$(id -u); gid=$(id -g); docker run -d --name {CAPTURE} --user 0 --network host --cap-add NET_RAW "
                f"-v {REMOTE}:/work -w /work --entrypoint sh {IMAGE} -c "
                f"\"python3 sniff_network.py --seconds {seconds} --out flows.json; chown $uid:$gid flows.json\"")
    print(f"Capturing for {seconds} s. Analyse some articles on the board's page now, then run: "
          f"python scripts/app/network_log.py fetch")


def fetch(board: Board, out: str) -> None:
    with open(os.path.join(out, "context.json"), encoding="utf-8") as f:
        context = json.load(f)
    board.shell(f"docker wait {CAPTURE} >/dev/null && docker rm {CAPTURE} >/dev/null")
    board.run("pull", f"{REMOTE}/flows.json", os.path.join(out, "flows.json"))
    with open(os.path.join(out, "flows.json"), encoding="utf-8") as f:
        capture = json.load(f)
    log = board_text(board, f"docker logs --since {int(capture['started'])} {APP_CONTAINER} 2>&1")
    summary = {"seconds": round(capture["seconds"]), **context,
               "analyses_logged": sum("Analysed" in line for line in log.splitlines()),
               "perspective_views_logged": sum("Perspective view" in line for line in log.splitlines()),
               **summarise(capture["flows"], context["app_ip"], context["board_ips"])}
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["start", "fetch"])
    parser.add_argument("--seconds", type=int, default=120)
    parser.add_argument("--adb", help="path to adb (default: PATH, then Arduino App Lab's copy)")
    parser.add_argument("--serial", help="adb serial number, if more than one device is connected")
    parser.add_argument("--out", default=os.path.join(ROOT, "outputs", "app", "network_log"))
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    board = Board(find_adb(args.adb), args.serial)
    if args.action == "start":
        start(board, args.seconds, args.out)
    else:
        fetch(board, args.out)


if __name__ == "__main__":
    main()
