// Bias Lens on the LED matrix: "Lens", a small eye-and-magnifier character, drawn in 8 levels of
// brightness and computed from the clock about 50 times a second, so motion is smooth.
//
//   start-up   a dot grows into a magnifying glass, a glint sweeps it, and it melts into the eye
//   idle       the eye glides its pupil around, blinks smoothly (now and then twice) and breathes
//   busy       the magnifier drifts over two scrolling lines of text, brightening what it reads
//   result     the heat strip rises column by column, brighter where more readers would flag;
//              the "most readers" columns pulse; a smile means nothing was flagged
//   pairing    the pairing code glides past, with three dots rippling under it
//   paired     a beating heart (lub-dub), then a tick that draws itself
//   error      a sad face, with a falling tear
//
// Every change of mode crossfades from the last picture. Python calls show_idle, show_busy,
// show_levels, show_code, show_paired and show_error over the Bridge. Those calls only leave a
// request; loop() draws without blocking.

#include <Arduino_LED_Matrix.h>
#include <Arduino_RouterBridge.h>
#include <math.h>

const int ROWS = 8;
const int COLS = 13;
const unsigned long FRAME_MS = 20;    // about 50 frames a second
const unsigned long FADE_MS = 220;    // crossfade between modes
const unsigned long RESULT_MS = 120000UL;  // results and pairing codes last 2 minutes
typedef const char *Frame[ROWS];

Arduino_LED_Matrix matrix;

enum Mode { BOOT, IDLE, BUSY, RESULT, CODE, PAIRED, SAD };
Mode mode = BOOT;
unsigned long modeStart = 0;
unsigned long lastFrame = 0;
int levels[COLS];
String code = "";

float canvas[ROWS][COLS];   // the picture being drawn, 0 to 1
float shown[ROWS][COLS];    // the last picture sent to the matrix
float fadeFrom[ROWS][COLS]; // the picture a crossfade starts from
uint8_t out[ROWS * COLS];   // 0 to 7, for matrix.draw

// Requests from the Bridge, picked up by loop().
volatile bool requested = false;
volatile Mode requestedMode = IDLE;
String requestedText = "";

// ------------------------------------------------------------------ pictures
// '#' full brightness, '+' half, '-' a faint glow, '.' off.

const Frame EYE_SHELL = {
  ".............",
  "....#####....",
  "..##.....##..",
  ".#.........#.",
  ".#.........#.",
  "..##.....##..",
  "....#####....",
  ".............",
};
const Frame EYE_WHITE = {
  ".............",
  ".............",
  "....-----....",
  "..---------..",
  "..---------..",
  "....-----....",
  ".............",
  ".............",
};
const Frame EYE_CLOSED = {
  ".............",
  ".............",
  ".............",
  ".#.........#.",
  "..#########..",
  ".............",
  ".............",
  ".............",
};
const Frame SMILE = {
  ".............",
  "...##...##...",
  "...##...##...",
  ".............",
  "..#.......#..",
  "...#.....#...",
  "....#####....",
  ".............",
};
const Frame SMILE_BLINK = {
  ".............",
  ".............",
  "...##...##...",
  ".............",
  "..#.......#..",
  "...#.....#...",
  "....#####....",
  ".............",
};
const Frame SAD_FACE = {
  ".............",
  "...##...##...",
  "...##...##...",
  ".............",
  ".............",
  "....#####....",
  "...#.....#...",
  "..#.......#..",
};
const Frame HEART_BIG = {
  ".............",
  "...##...##...",
  "..####.####..",
  "..#########..",
  "...#######...",
  "....#####....",
  ".....###.....",
  "......#......",
};
const Frame HEART_SMALL = {
  ".............",
  ".............",
  "....#...#....",
  "...###.###...",
  "...#######...",
  "....#####....",
  ".....###.....",
  "......#......",
};

// The tick, in the order it is drawn: row, column.
const int TICK[][2] = {{4, 2}, {5, 3}, {6, 4}, {6, 5}, {5, 6}, {4, 7}, {3, 8}, {2, 9}, {1, 10}};
const int TICK_LEN = 9;

// 3 x 5 digits for the pairing code; bit 2 is the left column.
const uint8_t DIGITS[10][5] = {
  {7, 5, 5, 5, 7}, {2, 6, 2, 2, 7}, {7, 1, 7, 4, 7}, {7, 1, 7, 1, 7}, {5, 5, 7, 1, 1},
  {7, 4, 7, 1, 7}, {7, 4, 7, 5, 7}, {7, 1, 2, 2, 2}, {7, 5, 7, 5, 7}, {7, 5, 7, 1, 7},
};

// Two lines of "text" for the reading animation.
const char *TEXT_A = "##.###.#.####.##.###..#.##.####.#";
const char *TEXT_B = "#.####.##.#.###.####.#.##.#.###.";

// ------------------------------------------------------------------ easing and drawing

float clamp01(float v) { return v < 0 ? 0 : (v > 1 ? 1 : v); }
float smooth(float x) { x = clamp01(x); return x * x * (3 - 2 * x); }
float easeOut(float x) { x = clamp01(x); return 1 - (1 - x) * (1 - x); }
float easeIn(float x) { x = clamp01(x); return x * x; }
float easeOutBack(float x) {  // rises past the target, then settles
  x = clamp01(x);
  const float c1 = 1.5f, c3 = c1 + 1;
  float d = x - 1;
  return 1 + c3 * d * d * d + c1 * d * d;
}
float wrap(float x, float m) { return x - m * floorf(x / m); }  // x modulo m, for floats

// Sine and cosine computed here: the firmware's sinf and cosf stall the board's Bridge.
const float TAU = 6.2831853f;
float fsin(float x) {
  x -= TAU * floorf(x / TAU + 0.5f);  // to -pi .. pi
  float y = 1.2732395f * x - 0.4052847f * x * fabsf(x);  // a parabola, then one refinement
  return 0.225f * (y * fabsf(y) - y) + y;
}
float fcos(float x) { return fsin(x + 1.5707963f); }

float bump(float t, float start, float length) {  // 0, up to 1 and back to 0
  float x = (t - start) / length;
  return (x <= 0 || x >= 1) ? 0 : fsin(x * 3.1415927f);
}

void clearCanvas() { memset(canvas, 0, sizeof(canvas)); }

void add(int r, int c, float v) {
  if (r >= 0 && r < ROWS && c >= 0 && c < COLS) canvas[r][c] = clamp01(canvas[r][c] + v);
}

void lighten(int r, int c, float v) {
  if (r >= 0 && r < ROWS && c >= 0 && c < COLS && v > canvas[r][c]) canvas[r][c] = clamp01(v);
}

float art(const Frame f, int r, int c) {
  char ch = f[r][c];
  return ch == '#' ? 1.0f : ch == '+' ? 0.55f : ch == '-' ? 0.16f : 0.0f;
}

void drawArt(const Frame f, float v) {
  for (int r = 0; r < ROWS; r++)
    for (int c = 0; c < COLS; c++) add(r, c, art(f, r, c) * v);
}

// Two pictures blended: mix 0 is all of a, 1 all of b.
void drawBlend(const Frame a, const Frame b, float mix, float v) {
  for (int r = 0; r < ROWS; r++)
    for (int c = 0; c < COLS; c++) add(r, c, (art(a, r, c) * (1 - mix) + art(b, r, c) * mix) * v);
}

// A picture squashed toward row cy (open 1: as drawn, 0: a line), with soft edges between rows.
void drawSquashed(const Frame f, float v, float open, float cy) {
  for (int r = 0; r < ROWS; r++)
    for (int c = 0; c < COLS; c++) {
      float a = art(f, r, c) * v;
      if (a <= 0) continue;
      float y = cy + (r - cy) * open;
      int y0 = (int)floorf(y);
      float fy = y - y0;
      add(y0, c, a * (1 - fy));
      add(y0 + 1, c, a * fy);
    }
}

// A ring of radius radius around (cx, cy), soft over width pixels.
void ring(float cx, float cy, float radius, float width, float v) {
  for (int r = 0; r < ROWS; r++)
    for (int c = 0; c < COLS; c++) {
      float d = hypotf(c - cx, r - cy);
      lighten(r, c, clamp01(1 - fabsf(d - radius) / width) * v);
    }
}

// A soft line from (x0, y0) to (x1, y1).
void segment(float x0, float y0, float x1, float y1, float width, float v) {
  float dx = x1 - x0, dy = y1 - y0, len2 = dx * dx + dy * dy;
  for (int r = 0; r < ROWS; r++)
    for (int c = 0; c < COLS; c++) {
      float t = len2 > 0 ? clamp01(((c - x0) * dx + (r - y0) * dy) / len2) : 0;
      float d = hypotf(c - (x0 + t * dx), r - (y0 + t * dy));
      lighten(r, c, clamp01(1 - d / width) * v);
    }
}

// The pupil: a 3 x 2 block whose left edge is at x (pixel centres are whole numbers), drawn
// with partial brightness where it covers part of a pixel, so it glides between columns.
void pupil(float x, float v, float open, float cy) {
  for (int c = 0; c < COLS; c++) {
    float cover = clamp01(fminf(c + 0.5f, x + 3) - fmaxf(c - 0.5f, x));
    if (cover <= 0) continue;
    for (int r = 3; r <= 4; r++) {
      float y = cy + (r - cy) * open;
      int y0 = (int)floorf(y);
      float fy = y - y0;
      add(y0, c, cover * v * (1 - fy));
      add(y0 + 1, c, cover * v * fy);
    }
  }
}

// The eye, with its pupil offset by look columns and its lids open by open (0 to 1).
void drawEye(float look, float open, float glow) {
  const float cy = 3.5f;
  float closedMix = open < 0.35f ? (0.35f - open) / 0.35f : 0;
  float o = fmaxf(open, 0.02f);
  drawSquashed(EYE_WHITE, glow, o, cy);
  drawSquashed(EYE_SHELL, 0.55f * (1 - closedMix), o, cy);
  pupil(4.5f + look, 1.0f * (1 - closedMix * 0.8f), o, cy);
  if (closedMix > 0) drawArt(EYE_CLOSED, 0.55f * closedMix);
}

// How open the lids are, tb milliseconds after a blink starts.
float blinkOpen(float tb) {
  if (tb < 0 || tb >= 250) return 1;
  if (tb < 80) return 1 - easeIn(tb / 80);
  if (tb < 120) return 0;
  return easeOut((tb - 120) / 130);
}

// Send the canvas to the matrix, crossfading from the previous mode's picture for FADE_MS.
void present(unsigned long now) {
  float fade = clamp01((float)(now - modeStart) / FADE_MS);
  float mix = smooth(fade);
  for (int r = 0; r < ROWS; r++)
    for (int c = 0; c < COLS; c++) {
      float v = fadeFrom[r][c] * (1 - mix) + canvas[r][c] * mix;
      shown[r][c] = v;
      out[r * COLS + c] = (uint8_t)(clamp01(v) * 7 + 0.5f);
    }
  matrix.draw(out);
}

// ------------------------------------------------------------------ modes

void enter(Mode m) {
  memcpy(fadeFrom, shown, sizeof(shown));
  mode = m;
  modeStart = millis();
}

void drawBoot(float t) {
  const float cx = 4.5f, cy = 3.0f, radius = 2.5f;
  if (t < 450) {  // a dot glows up
    float g = smooth(t / 450);
    ring(cx, cy, 0, 0.6f + 0.9f * g, g);
  } else if (t < 1000) {  // it opens into a ring
    float g = smooth((t - 450) / 550);
    ring(cx, cy, radius * g, 1.0f, 0.85f);
  } else if (t < 1900) {  // the handle draws in, then a glint sweeps the lens
    ring(cx, cy, radius, 1.0f, 0.85f);
    float h = smooth((t - 1000) / 350);
    segment(cx + 1.8f, cy + 1.8f, cx + 1.8f + 3.2f * h, cy + 1.8f + 3.2f * h, 0.9f, 0.85f);
    float sweep = (t - 1350) / 450;  // a bright diagonal band crossing the lens
    if (sweep > 0 && sweep < 1) {
      float band = -3 + 6 * sweep;
      for (int r = 0; r < ROWS; r++)
        for (int c = 0; c < COLS; c++) {
          float d = hypotf(c - cx, r - cy);
          if (d < radius) lighten(r, c, clamp01(1 - fabsf((c - cx) + (r - cy) - band) / 1.2f) * 0.9f);
        }
    }
  } else {  // the lens melts into the opening eye
    float g = smooth((t - 1900) / 600);
    ring(cx, cy, radius, 1.0f, 0.85f * (1 - g));
    segment(cx + 1.8f, cy + 1.8f, cx + 5.0f, cy + 5.0f, 0.9f, 0.85f * (1 - g));
    drawEye(0, g, 0.16f * g);
    if (t > 2500) enter(IDLE);
  }
}

// The idle eye's script: how long each part lasts, where the pupil looks, and whether it starts
// with a blink (2: a double blink every third time round).
struct Glance { uint16_t ms; int8_t look; uint8_t blink; };
const Glance GLANCES[] = {
  {2400, 0, 0}, {1000, 0, 1}, {1100, -2, 0}, {600, 0, 0}, {1100, 2, 0},
  {1600, 0, 0}, {450, 1, 0}, {450, -1, 0}, {1400, 0, 2},
};
const int GLANCE_COUNT = sizeof(GLANCES) / sizeof(GLANCES[0]);

void drawIdle(float t) {
  unsigned long total = 0;
  for (int i = 0; i < GLANCE_COUNT; i++) total += GLANCES[i].ms;
  unsigned long lap = (unsigned long)t / total;
  float within = wrap(t, (float)total);
  int i = 0;
  float start = 0;
  while (i < GLANCE_COUNT - 1 && within >= start + GLANCES[i].ms) start += GLANCES[i++].ms;
  float from = GLANCES[(i + GLANCE_COUNT - 1) % GLANCE_COUNT].look;
  float look = from + (GLANCES[i].look - from) * smooth((within - start) / 280);
  float open = 1;
  if (GLANCES[i].blink) {
    open = blinkOpen(within - start);
    if (GLANCES[i].blink == 2 && lap % 3 == 2) open = fminf(open, blinkOpen(within - start - 320));
  }
  float glow = 0.13f + 0.05f * fsin(t / 4000.0f * TAU);  // the eye breathes
  drawEye(look, open, glow);
}

void drawBusy(float t) {
  const float radius = 2.4f, cy = 3.0f;
  float lx = 2.5f + (COLS - 6) * (0.5f - 0.5f * fcos(t / 2600.0f * TAU));  // drifts back and forth
  float shift = t / 140.0f;  // the text scrolls left, smoothly
  int whole = (int)floorf(shift);
  float frac = shift - whole;
  int na = strlen(TEXT_A), nb = strlen(TEXT_B);
  for (int c = 0; c < COLS; c++) {
    float d2 = hypotf(c - lx, 2 - cy), d4 = hypotf(c - lx, 4 - cy);
    float a = (TEXT_A[(c + whole) % na] == '#') * (1 - frac) + (TEXT_A[(c + whole + 1) % na] == '#') * frac;
    float b = (TEXT_B[(c + whole + 7) % nb] == '#') * (1 - frac) + (TEXT_B[(c + whole + 8) % nb] == '#') * frac;
    add(2, c, a * (0.18f + 0.82f * clamp01((radius - d2) * 1.2f)));  // brighter under the lens
    add(4, c, b * (0.18f + 0.82f * clamp01((radius - d4) * 1.2f)));
  }
  ring(lx, cy, radius, 0.9f, 0.75f);
  segment(lx + 1.7f, cy + 1.7f, lx + 3.4f, cy + 3.9f, 0.85f, 0.75f);
}

void drawResult(float t) {
  int tallest = 0;
  for (int c = 0; c < COLS; c++) tallest = max(tallest, levels[c]);
  if (tallest == 0) {  // nothing flagged: a smile that blinks now and then
    float blink = bump(wrap(t, 4200), 2600, 260);
    drawBlend(SMILE, SMILE_BLINK, blink, smooth(t / 400));
    return;
  }
  const float stagger = 45, rise = 450;
  float settled = (COLS - 1) * stagger + rise;
  for (int c = 0; c < COLS; c++) {
    int level = levels[c];
    float heat = 0.35f + 0.65f * level / 8.0f;  // brighter where more readers would flag
    if (level >= 5 && t > settled && t < settled + 2700)  // "most readers" columns pulse three times
      heat *= 0.7f + 0.3f * fcos((t - settled) / 900.0f * TAU);
    float h = level * easeOutBack((t - c * stagger) / rise);
    if (level == 0) { add(ROWS - 1, c, 0.12f * smooth(t / 300)); continue; }  // a faint base: an empty stretch
    for (int r = 0; r < ROWS; r++) {
      float fromBottom = ROWS - r;  // 1 for the bottom row
      add(r, c, clamp01(h - (fromBottom - 1)) * heat);
    }
  }
}

// Pairing code as a marquee: 13 blank columns, the digits "123 456", 13 blank columns.
int codeColumns() { return COLS + (int)code.length() * 4 + 1 + COLS; }

uint8_t codeColumn(int index) {
  index -= COLS;
  if (index < 0) return 0;
  for (int i = 0; i < (int)code.length(); i++) {
    int width = (i == 2) ? 5 : 4;  // a wider gap after the third digit
    if (index < width) {
      char ch = code[i];
      if (index >= 3 || ch < '0' || ch > '9') return 0;
      uint8_t column = 0;
      for (int r = 0; r < 5; r++)
        if (DIGITS[ch - '0'][r] & (4 >> index)) column |= (1 << r);
      return column;
    }
    index -= width;
  }
  return 0;
}

void drawCode(float t) {
  float offset = wrap(t * 0.0115f, (float)codeColumns());  // about 11 columns a second
  int whole = (int)floorf(offset);
  float frac = offset - whole;
  int n = codeColumns();
  for (int c = 0; c < COLS; c++) {
    uint8_t a = codeColumn((whole + c) % n), b = codeColumn((whole + c + 1) % n);
    for (int r = 0; r < 5; r++)
      add(r + 1, c, ((a >> r) & 1) * (1 - frac) + ((b >> r) & 1) * frac);
  }
  for (int i = 0; i < 3; i++)  // three dots under the code ripple: "type me in"
    add(7, 4 + 2 * i, 0.25f + 0.75f * fmaxf(0, fcos((t / 900.0f - i / 3.0f) * TAU)));
}

void drawPaired(float t) {
  if (t < 2700) {  // three heartbeats: lub-dub
    float beat = wrap(t, 900);
    float swell = fmaxf(bump(beat, 0, 170), 0.75f * bump(beat, 230, 170));
    drawBlend(HEART_SMALL, HEART_BIG, swell, 1.0f);
    return;
  }
  float s = t - 2700;  // then the tick draws itself in
  for (int i = 0; i < TICK_LEN; i++) add(TICK[i][0], TICK[i][1], smooth((s - i * 45) / 90));
  if (s > 2000) enter(IDLE);
}

void drawSad(float t) {
  drawArt(SAD_FACE, smooth(t / 350));
  float fall = (t - 700) / 900;  // a tear falls from under the left eye
  if (fall > 0 && fall < 1) {
    float y = 3 + 4.4f * easeIn(fall);
    int y0 = (int)floorf(y);
    float fy = y - y0, v = 0.8f * (1 - 0.5f * fall);
    add(y0, 3, v * (1 - fy));
    add(y0 + 1, 3, v * fy);
  }
  if (t > 2800) enter(IDLE);
}

// ------------------------------------------------------------------ Bridge calls

void request(Mode m, String text) {
  requestedText = text;
  requestedMode = m;
  requested = true;
}

void show_idle() { request(IDLE, ""); }
void show_busy() { request(BUSY, ""); }
void show_levels(String text) { request(RESULT, text); }  // 13 digits, '0' to '8'
void show_code(String text) { request(CODE, text); }      // the pairing code, digits only
void show_paired() { request(PAIRED, ""); }
void show_error() { request(SAD, ""); }

void setup() {
  matrix.begin();
  matrix.setGrayscaleBits(3);  // 8 levels: 0 off to 7 full
  memset(shown, 0, sizeof(shown));
  Bridge.begin();
  Bridge.provide("show_idle", show_idle);
  Bridge.provide("show_busy", show_busy);
  Bridge.provide("show_levels", show_levels);
  Bridge.provide("show_code", show_code);
  Bridge.provide("show_paired", show_paired);
  Bridge.provide("show_error", show_error);
  enter(BOOT);
}

void loop() {
  if (requested) {
    requested = false;
    Mode m = requestedMode;
    String text = requestedText;
    if (m == RESULT) {
      for (int c = 0; c < COLS; c++)
        levels[c] = c < (int)text.length() ? constrain(text[c] - '0', 0, ROWS) : 0;
    }
    if (m == CODE) code = text;
    if (!(m == IDLE && mode == BOOT)) enter(m);  // let the start-up animation finish
  }
  unsigned long now = millis();
  if (now - lastFrame < FRAME_MS) {
    delay(2);
    return;
  }
  lastFrame = now;
  float t = (float)(now - modeStart);
  clearCanvas();
  switch (mode) {
    case BOOT: drawBoot(t); break;
    case IDLE: drawIdle(t); break;
    case BUSY: drawBusy(t); break;
    case RESULT:
      drawResult(t);
      if (t > RESULT_MS) enter(IDLE);  // back to the eye after 2 minutes
      break;
    case CODE:
      drawCode(t);
      if (t > RESULT_MS) enter(IDLE);
      break;
    case PAIRED: drawPaired(t); break;
    case SAD: drawSad(t); break;
  }
  present(now);
}
