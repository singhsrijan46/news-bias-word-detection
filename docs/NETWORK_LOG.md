# Network log: does any article text leave the board?

Bias Lens promises that text is analysed on the board and sent nowhere else. The proposal names a network log of the device as the evidence for that promise, under the privacy requirement of Trustworthy AI. This is that log.

## Method

`scripts/app/network_log.py` records every IP packet on every network interface of the board while Bias Lens is in use.
- **What is recorded:** the board side, `scripts/app/sniff_network.py`, keeps each packet's interface, direction, protocol, addresses and ports, and counts packets and bytes. It never records contents.
- **How it sees everything:** it runs in App Lab's own Python image, with the host's network and the raw-socket capability. So it sees the Wi-Fi (`wlan0`), loopback, and Docker's bridges. The app's container has its own bridge network, at 172.18.0.2, so its traffic can be told apart from the rest of the board's.
- **Grouping:** every flow goes into one of the groups below. Anything the app's container sent anywhere other than its web server would land in "the app and elsewhere".
- **The workload:** during a 150-second capture on 2026-11-08, a paired browser had six texts analysed, three of 82 words and three of 820 words. It also asked for one liberal and conservative comparison. The app logged all seven, as counts. The browser reached the board over USB, through adb's port forwarding, so the requests themselves did not cross the Wi-Fi.

A static check backs this up: the app's Python code opens no network connections. Its one client, in `explainer.py`, talks to the board's own llama.cpp server (`docs/EXPLANATIONS.md`) through a Unix socket: a file in a private folder, which carries no IP traffic. The same client can also use a local port, but only the evaluation script does that, on the PC.

## Results

| Group | Flows | Packets | What it was |
| --- | --- | --- | --- |
| The app's web server | 12 | 184 | Requests and answers on port 7000, through Docker's port forwarding |
| **The app and elsewhere** | **0** | **0** | Nothing |
| The app and the rest of the board | 0 | 0 | The LED matrix's Bridge did not use IP |
| **The board and the internet** | **0** | **0** | Nothing |
| The board and the home network | 25 | 70 | Device discovery: `.local` name lookups (mDNS, port 5353) that the board answered, IPv6 neighbour discovery and multicast group reports, mostly from other devices |
| Inside the board | 37 | 4,119 | Loopback and Docker's bridge: the forwarded requests on their way to the app, and the board's own services |

- **The app talked to no one but its clients.** Its container exchanged packets only on its web server's port. It made no DNS lookups and sent no telemetry or anything else.
- **The board talked to nothing beyond the home network** during the capture.
- **The home-network traffic carries no text.** It is device discovery, which runs whether or not the app is in use. The board's own answers give its name and address.

**A second capture, with the language model.** After the explainer moved to llama.cpp's server (`docs/EXPLANATIONS.md`), a second 150-second capture on the same day covered:
- **The workload:** a restart of the app, one analysis and three explanations. The first explanation also started llama.cpp's server, and the app logged all four requests, as counts.
- **The same picture:** the app's web server had 20 flows and 270 packets. Nothing went between the app and anywhere else, and nothing between the board and the internet. The home network carried device discovery (25 flows, 57 packets), and inside the board there were 45 flows and 5,592 packets.

So llama.cpp's server, inside the app's container, used no network. The app reaches it through a Unix socket.

## Run it

With the app running on the board (`docs/BIAS_LENS_APP.md`):

```bash
python scripts/app/network_log.py start --seconds 150    # then analyse some articles on the board's page
python scripts/app/network_log.py fetch                  # outputs/app/network_log/summary.json
```

## Limits

- **One window.** The log covers 150 seconds. The board's own system services, such as App Lab's update checks, may go online at other times. The claim is about the app: its container, which made no outside contact.
- **Over USB.** A phone on the Wi-Fi would add its requests and answers to the app's web server group, on `wlan0`. Those are encrypted end to end (`docs/BIAS_LENS_CLIENTS.md`).
- **The page itself.** The page and the bookmark load nothing from outside the board, so a browser has nothing to fetch elsewhere either. The extension's own traffic, in the reader's browser, is not part of this log.
