# Bias Lens

Paste a news article and Bias Lens marks the words that readers would call biased, with how many readers in ten would flag each one. Words that some readers flag and others do not are shown as contested.

Everything runs on the UNO Q:
- The model runs on the board's CPU.
- The web page is served to phones and laptops on the home network at `http://<board name>.local:7000`.
- The LED matrix shows a heat strip of the last article. Each of its 13 columns is a stretch of the article, and taller means more strongly flagged.

No text leaves the board.

The scores come from MBIC: about ten US crowd workers per sentence, reading 1,700 news sentences from 2017 to 2020. A highlight is a prompt to look again, not a ruling.

Set up and deploy from the project repository with `scripts/app/deploy_app.py`, which also adds the model. See `docs/BIAS_LENS_APP.md` there.
