// Bias Lens on the LED matrix: "Lens", a small eye-and-magnifier character.
//
//   start-up   a dot grows into a magnifying glass, it glints, and an eye opens
//   idle       the eye looks around and blinks (now and then twice)
//   busy       the magnifier reads along a scrolling line of text
//   result     the heat strip rises bar by bar: one column per stretch of the article,
//              taller when more readers would flag its words; tall bars wink, and a
//              smile means nothing was flagged
//   pairing    the pairing code scrolls past, to type into a new device
//   paired     a beating heart, then a tick
//   error      a sad face
//
// Python calls show_idle, show_busy, show_levels, show_code, show_paired and show_error
// over the Bridge. Those calls only leave a request; loop() animates without blocking.

#include <Arduino_LED_Matrix.h>
#include <Arduino_RouterBridge.h>

const int ROWS = 8;
const int COLS = 13;
typedef const char *Frame[ROWS];

Arduino_LED_Matrix matrix;
uint8_t pixels[ROWS * COLS];

enum Mode { BOOT, IDLE, BUSY, RESULT, CODE, PAIRED, SAD };
Mode mode = BOOT;
unsigned long modeStart = 0;
unsigned long nextStep = 0;
int step = 0;
int levels[COLS];
String code = "";

// Requests from the Bridge, picked up by loop().
volatile bool requested = false;
volatile Mode requestedMode = IDLE;
String requestedText = "";

// ------------------------------------------------------------------ frames

const Frame EYE_CENTER = {
  ".............",
  "....#####....",
  "..##.....##..",
  ".#...###...#.",
  ".#...###...#.",
  "..##.....##..",
  "....#####....",
  ".............",
};
const Frame EYE_LEFT = {
  ".............",
  "....#####....",
  "..##.....##..",
  ".#.###.....#.",
  ".#.###.....#.",
  "..##.....##..",
  "....#####....",
  ".............",
};
const Frame EYE_RIGHT = {
  ".............",
  "....#####....",
  "..##.....##..",
  ".#.....###.#.",
  ".#.....###.#.",
  "..##.....##..",
  "....#####....",
  ".............",
};
const Frame EYE_HALF = {
  ".............",
  ".............",
  "..#########..",
  ".#...###...#.",
  "..#########..",
  ".............",
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
const Frame DOT = {
  ".............",
  ".............",
  ".............",
  "......#......",
  ".............",
  ".............",
  ".............",
  ".............",
};
const Frame RING = {
  ".............",
  ".............",
  ".....###.....",
  ".....#.#.....",
  ".....###.....",
  ".............",
  ".............",
  ".............",
};
const Frame MAGNIFIER = {
  "....###......",
  "...#...#.....",
  "..#.....#....",
  "..#.....#....",
  "..#.....#....",
  "...#...#.....",
  "....###.#....",
  ".........##..",
};
const Frame MAGNIFIER_GLINT = {
  "....###......",
  "...#...#.....",
  "..#.#...#....",
  "..#..#..#....",
  "..#.....#....",
  "...#...#.....",
  "....###.#....",
  ".........##..",
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
const Frame TICK = {
  ".............",
  "..........#..",
  ".........#...",
  "........#....",
  "..#....#.....",
  "...#..#......",
  "....##.......",
  ".............",
};

// 3 x 5 digits for the pairing code; bit 2 is the left column.
const uint8_t DIGITS[10][5] = {
  {7, 5, 5, 5, 7}, {2, 6, 2, 2, 7}, {7, 1, 7, 4, 7}, {7, 1, 7, 1, 7}, {5, 5, 7, 1, 1},
  {7, 4, 7, 1, 7}, {7, 4, 7, 5, 7}, {7, 1, 2, 2, 2}, {7, 5, 7, 5, 7}, {7, 5, 7, 1, 7},
};

// ------------------------------------------------------------------ drawing

void clearPixels() { memset(pixels, 0, sizeof(pixels)); }

void set(int r, int c) {
  if (r >= 0 && r < ROWS && c >= 0 && c < COLS) pixels[r * COLS + c] = 1;
}

void show() { matrix.loadPixels(pixels, ROWS * COLS); }

void drawFrame(const Frame frame) {
  for (int r = 0; r < ROWS; r++)
    for (int c = 0; c < COLS; c++) pixels[r * COLS + c] = frame[r][c] == '#' ? 1 : 0;
  show();
}

// The magnifier for "busy": a 5 x 5 lens with a handle, at column x, over a line of text.
void drawReading(int x, int textShift) {
  clearPixels();
  for (int c = 1; c <= 3; c++) { set(0, x + c); set(4, x + c); }
  for (int r = 1; r <= 3; r++) { set(r, x); set(r, x + 4); }
  set(5, x + 4);
  set(6, x + 5);
  const char *line = "###.##.####.#.###.##.#####.##.";
  int n = strlen(line);
  for (int c = 0; c < COLS; c++)
    if (line[(c + textShift) % n] == '#') set(7, c);
  show();
}

void drawBars(int height, int hideLevel) {
  clearPixels();
  for (int c = 0; c < COLS; c++) {
    if (levels[c] == hideLevel) continue;
    int h = min(levels[c], height);
    for (int r = ROWS - h; r < ROWS; r++) set(r, c);
  }
  show();
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

void drawCode(int offset) {
  clearPixels();
  for (int c = 0; c < COLS; c++) {
    uint8_t column = codeColumn(offset + c);
    for (int r = 0; r < 5; r++)
      if (column & (1 << r)) set(r + 1, c);
  }
  set(7, 4); set(7, 6); set(7, 8);  // three dots under the code: "type me in"
  show();
}

// ------------------------------------------------------------------ modes

void enter(Mode m) {
  mode = m;
  modeStart = millis();
  nextStep = modeStart;
  step = 0;
}

// Each animation returns how long to wait before its next step.
unsigned long animateBoot() {
  switch (step) {
    case 0: drawFrame(DOT); return 180;
    case 1: drawFrame(RING); return 180;
    case 2: drawFrame(MAGNIFIER); return 500;
    case 3: drawFrame(MAGNIFIER_GLINT); return 160;
    case 4: drawFrame(MAGNIFIER); return 420;
    case 5: drawFrame(EYE_CLOSED); return 160;
    case 6: drawFrame(EYE_HALF); return 120;
    default: enter(IDLE); return 0;
  }
}

unsigned long animateIdle() {
  // A loop of looking around and blinking; every third round the blink is doubled.
  static const int SCRIPT_LEN = 12;
  int round = step / SCRIPT_LEN;
  switch (step % SCRIPT_LEN) {
    case 0: drawFrame(EYE_CENTER); return 2400;
    case 1: drawFrame(EYE_HALF); return 60;
    case 2: drawFrame(EYE_CLOSED); return 110;
    case 3: drawFrame(EYE_HALF); return 60;
    case 4: if (round % 3 == 2) { drawFrame(EYE_CLOSED); return 110; } drawFrame(EYE_CENTER); return 900;
    case 5: drawFrame(EYE_LEFT); return 900;
    case 6: drawFrame(EYE_CENTER); return 500;
    case 7: drawFrame(EYE_RIGHT); return 900;
    case 8: drawFrame(EYE_CENTER); return 1600;
    case 9: drawFrame(EYE_RIGHT); return 300;
    case 10: drawFrame(EYE_LEFT); return 300;
    default: drawFrame(EYE_CENTER); return 1200;
  }
}

unsigned long animateBusy() {
  int travel = COLS - 6;  // the lens is 6 columns wide with its handle
  int phase = step % (2 * travel);
  int x = phase < travel ? phase : 2 * travel - phase;
  drawReading(x, step);
  return 85;
}

unsigned long animateResult() {
  int tallest = 0;
  for (int c = 0; c < COLS; c++) tallest = max(tallest, levels[c]);
  if (tallest == 0) {  // nothing flagged
    if (step == 0) { drawFrame(SMILE); return 4000; }
    if (step == 1) { drawFrame(EYE_CLOSED); return 150; }  // a wink, then hold the smile
    drawFrame(SMILE);
    return millis() - modeStart > 120000UL ? (enter(IDLE), 0) : 1000;
  }
  if (step < ROWS) { drawBars(step + 1, -1); return 45; }  // bars rise
  if (tallest >= 7 && step < ROWS + 6) {  // the tallest bars wink three times
    drawBars(ROWS, (step - ROWS) % 2 == 0 ? tallest : -1);
    return 170;
  }
  drawBars(ROWS, -1);
  return millis() - modeStart > 120000UL ? (enter(IDLE), 0) : 1000;  // back to the eye after 2 minutes
}

unsigned long animateCode() {
  drawCode(step % codeColumns());
  return millis() - modeStart > 120000UL ? (enter(IDLE), 0) : 110;  // pairing codes last 2 minutes
}

unsigned long animatePaired() {
  if (step < 6) { drawFrame(step % 2 == 0 ? HEART_BIG : HEART_SMALL); return step % 2 == 0 ? 260 : 180; }
  if (step == 6) { drawFrame(TICK); return 1400; }
  enter(IDLE);
  return 0;
}

unsigned long animateSad() {
  if (step == 0) { drawFrame(SAD_FACE); return 2500; }
  enter(IDLE);
  return 0;
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
  matrix.clear();
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
  if ((long)(now - nextStep) < 0) {
    delay(5);
    return;
  }
  unsigned long wait = 0;
  switch (mode) {
    case BOOT: wait = animateBoot(); break;
    case IDLE: wait = animateIdle(); break;
    case BUSY: wait = animateBusy(); break;
    case RESULT: wait = animateResult(); break;
    case CODE: wait = animateCode(); break;
    case PAIRED: wait = animatePaired(); break;
    case SAD: wait = animateSad(); break;
  }
  if (wait > 0) {
    step++;
    nextStep = now + wait;
  }
}
