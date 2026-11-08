#!/usr/bin/env python3
"""
Count every IP packet on every network interface of the board for a while, by flow.

This is the board side of network_log.py. It needs the host's network and the raw-socket
capability (docker run --network host --cap-add NET_RAW), and only Python's standard library.
It records addresses, ports, packet counts and sizes, never packet contents.

    python3 sniff_network.py --seconds 120 --out flows.json
"""

import argparse
import json
import socket
import struct
import time
from collections import defaultdict

ETH_P_ALL = 0x0003
PROTOCOLS = {0: "ipv6-hop-by-hop", 1: "icmp", 2: "igmp", 6: "tcp", 17: "udp", 58: "icmpv6"}
DIRECTIONS = {0: "in", 1: "in", 2: "in", 3: "seen", 4: "out"}  # Linux packet types: to us, broadcast, multicast, other host, outgoing


def parse(frame: bytes):
    """(protocol, source, destination, source port, destination port) of an Ethernet frame's IP packet, or None."""
    if len(frame) < 14:
        return None
    ethertype = struct.unpack("!H", frame[12:14])[0]
    if ethertype == 0x0800 and len(frame) >= 34:
        proto = frame[23]
        src, dst = socket.inet_ntoa(frame[26:30]), socket.inet_ntoa(frame[30:34])
        offset = 14 + (frame[14] & 0x0F) * 4
    elif ethertype == 0x86DD and len(frame) >= 54:
        proto = frame[20]
        src = socket.inet_ntop(socket.AF_INET6, frame[22:38])
        dst = socket.inet_ntop(socket.AF_INET6, frame[38:54])
        offset = 54
    else:
        return None
    sport = dport = None
    if proto in (6, 17) and len(frame) >= offset + 4:
        sport, dport = struct.unpack("!HH", frame[offset:offset + 4])
    return PROTOCOLS.get(proto, str(proto)), src, dst, sport, dport


def capture(seconds: float) -> dict:
    sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(ETH_P_ALL))
    sock.settimeout(0.5)
    flows = defaultdict(lambda: [0, 0])
    started = time.time()
    while time.time() < started + seconds:
        try:
            frame, address = sock.recvfrom(65535)
        except socket.timeout:
            continue
        parsed = parse(frame)
        if parsed is not None:
            key = (address[0], DIRECTIONS.get(address[2], str(address[2]))) + parsed
            flows[key][0] += 1
            flows[key][1] += len(frame)
    names = ("interface", "direction", "protocol", "src", "dst", "sport", "dport")
    return {"started": started, "seconds": time.time() - started,
            "flows": [{**dict(zip(names, key)), "packets": n, "bytes": size} for key, (n, size) in flows.items()]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    result = capture(args.seconds)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f)
    print(f"{len(result['flows'])} flows in {result['seconds']:.0f} s")


if __name__ == "__main__":
    main()
