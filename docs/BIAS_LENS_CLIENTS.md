# Reading with Bias Lens: the browser extension and the phone bookmark

Bias Lens marks biased words directly on the news sites you read. The board does the analysis; a small client in your browser finds the article, sends its text to the board over an encrypted channel, and marks the words in place.

| Where you read | Client | How it works |
| --- | --- | --- |
| Chrome or Edge on a laptop | The extension, `clients/extension` | Automatic on news sites; a button for any other page |
| Chrome on Android (no extensions there) | The phone bookmark | One tap per article |
| Any browser | The board's own page | Paste text |

## Why the board does not read your traffic

Almost every news site uses HTTPS, so the board would see only encrypted bytes and site names. To read the pages it would have to break HTTPS for all your browsing, by installing its own root certificate on your devices as a man-in-the-middle. Then a bug or break-in on the board would expose your banking and email as well. Your browser already has the decrypted page, so the clients send the board only the article text, and only from news pages.

## The extension (Chrome, Edge)

1. **Build it:** run `python scripts/clients/build_extension.py`. This creates `outputs/clients/bias-lens-extension/`.
2. **Load it:** open `chrome://extensions` (or `edge://extensions`), switch on **Developer mode**, choose **Load unpacked**, and pick that folder.
3. **Connect it:** click the Bias Lens icon and type the board's address, as shown on the board's page, for example `http://192.168.1.7:7000`. Click **Connect**. The browser asks for permission to reach exactly that address and nothing else.
4. **Pair it:** click **Show code**. Six digits scroll across the board's LED matrix; type them in and click **Pair**. The matrix shows a heart.

After that, articles on the listed news sites, such as nytimes.com, foxnews.com, reuters.com, theguardian.com and ndtv.com (the full list is in `manifest.json`), are marked automatically about a second and a half after they load. A card in the corner shows the heat strip and can hide the highlights. For any other page, click the icon and choose **Analyze this page**. You can pause the extension on a site, turn off automatic highlighting, or unpair, all from the same window.

## The phone bookmark (Chrome on Android)

Chrome on Android does not support extensions, so a bookmark does the extension's job, one tap per article.

1. **Pair the phone:** on the phone, open the board's page (`http://<board address>:7000`) and pair it with the code from the LED matrix.
2. **Copy the bookmark:** on the same page, open **Use Bias Lens while you read** and tap **Copy the bookmark**.
3. **Create the bookmark:** bookmark any page, then edit that bookmark. Name it `Bias Lens` and paste the copied text into its address.
4. **Use it:** on a news article, tap the address bar, type `Bias Lens`, and tap the bookmark.
   - A board tab opens, reads the article, sends the result back to the article, and closes.
   - The highlights appear on the article. Tap a marked word to see how many readers would flag it.

If a site prevents the result from coming back, the board tab shows the marked-up article itself instead. Some sites with strict security rules (a Content Security Policy) may block bookmarks altogether; on those, paste the text into the board's page.

## Security

- **Pairing.** Each device pairs once, with a 6-digit code that only the board's LED matrix shows. Under the hood is a password-authenticated key exchange in the style of CPace, over ristretto255 (`app/bias-lens/python/security.py`, `clients/shared/biaslens-client.js`).
  - **Guessing the code:** someone on your network who cannot see the board gets one guess per 2-minute pairing session, and the board allows at most 30 sessions an hour.
  - **Eavesdropping:** someone who records the exchange cannot test codes against it afterwards.
  - **Result:** each paired device ends up with its own 256-bit key.
- **Requests.** Every article is sent encrypted and authenticated with AES-256-GCM under the device's key. The device, a timestamp and the request's nonce are bound in, so a request cannot be replayed or altered, and answers come back encrypted the same way. Unpaired devices are refused.
- **Nothing kept.** The board stores no text and no history: only the list of paired devices. Any device can be unpaired from its client.
- **Tested.** `tests/test_security.py` covers wrong codes, expired sessions, replays, tampering and unknown devices. It also checks that the JavaScript client and the Python board derive identical keys.
- **Limits.**
  - **Unreviewed design.** The protocol is built from standard parts, but nobody outside the project has reviewed how they are combined.
  - **The board's page is the weak spot.** The page, the phone's half of the phone path, is served over plain HTTP. An attacker who can rewrite traffic on your home network could therefore alter that page's code. The extension's code is installed on the device, so this does not apply to it.
  - **The fix for strict setups:** serve the board over HTTPS, for example inside a WireGuard or Tailscale network, which also allows reading away from home.
  - **For testing only:** `data/settings.json` on the board can set `debug_write_code`, which also writes the pairing code to a file that only someone with access to the board itself can read. It is off by default, and should stay off.

## What the highlights mean

The same as on the board's page (`docs/BIAS_LENS_APP.md`).
- **Red:** most readers would flag the word, 6 or more in 10.
- **Amber with a dashed underline:** readers split on it.

The model learned from MBIC's US crowd workers rating US political news, so on other countries' news it reflects those readers' sensitivities, not local ones.
