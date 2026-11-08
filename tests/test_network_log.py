import importlib.util
import os
import socket
import struct

APP_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "app")


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(APP_SCRIPTS, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sniff = load("sniff_network")
network_log = load("network_log")


def udp_frame(src, dst, sport, dport):
    ethernet = b"\x00" * 12 + struct.pack("!H", 0x0800)
    ip = struct.pack("!BBHHHBBH4s4s", 0x45, 0, 28, 0, 0, 64, 17, 0, socket.inet_aton(src), socket.inet_aton(dst))
    return ethernet + ip + struct.pack("!HHHH", sport, dport, 8, 0)


def test_frames_are_reduced_to_addresses_and_ports():
    assert sniff.parse(udp_frame("192.168.1.13", "224.0.0.251", 5353, 5353)) == (
        "udp", "192.168.1.13", "224.0.0.251", 5353, 5353)
    assert sniff.parse(b"\x00" * 12 + struct.pack("!H", 0x0806) + b"\x00" * 28) is None  # ARP: not IP


APP, BOARD = "172.18.0.2", ["192.168.1.7", "172.18.0.1"]


def flow(interface, direction, src, dst, sport=None, dport=None, packets=1):
    return {"interface": interface, "direction": direction, "protocol": "tcp", "src": src, "dst": dst,
            "sport": sport, "dport": dport, "packets": packets, "bytes": 60 * packets}


def test_flows_are_grouped_by_who_talks_to_whom():
    group = lambda f: network_log.group(f, APP, BOARD)  # noqa: E731
    assert group(flow("br-1", "out", "172.18.0.1", APP, 33334, 7000)) == "app's web server"
    assert group(flow("br-1", "in", APP, "172.18.0.1", 41000, 9000)) == "app and the board"
    assert group(flow("br-1", "in", APP, "93.184.216.34", 41000, 443)) == "app and elsewhere"
    assert group(flow("wlan0", "out", "192.168.1.7", "93.184.216.34", 40000, 443)) == "board, internet"
    assert group(flow("wlan0", "in", "192.168.1.13", "224.0.0.251", 5353, 5353)) == "board, home network"
    assert group(flow("lo", "out", "127.0.0.1", "127.0.0.1", 50000, 8800)) == "inside the board"


def test_summary_lists_anything_the_app_sent_elsewhere():
    flows = [flow("br-1", "out", "172.18.0.1", APP, 33334, 7000, packets=10),
             flow("br-1", "in", APP, "93.184.216.34", 41000, 443, packets=3)]
    summary = network_log.summarise(flows, APP, BOARD)
    assert summary["groups"]["app's web server"]["packets"] == 10
    assert [f["dst"] for f in summary["app_and_elsewhere"]] == ["93.184.216.34"]
    assert summary["board_internet"] == []
