// Bias Lens: the LED matrix shows the last analysed article as a "heat strip".
//
// The article is cut into 13 stretches, one per column. A column's height is how
// strongly its most-flagged word is flagged: empty when nothing is flagged, all 8
// rows when every reader would flag it. While an article is being analysed, a line
// sweeps across the matrix; when idle, it shows a magnifying glass.
//
// The Python side calls show_idle, show_busy and show_levels over the Bridge.

#include <Arduino_LED_Matrix.h>
#include <Arduino_RouterBridge.h>

const int ROWS = 8;
const int COLS = 13;

Arduino_LED_Matrix matrix;
uint8_t pixels[ROWS * COLS];
volatile bool busy = false;
int sweep = 0;

// 8 x 13 magnifying glass, one string per row.
const char *IDLE_ICON[ROWS] = {
  "....###......",
  "...#...#.....",
  "..#.....#....",
  "..#.....#....",
  "..#.....#....",
  "...#...#.....",
  "....###.#....",
  ".........##..",
};

void render() {
  matrix.loadPixels(pixels, ROWS * COLS);
}

void drawIdle() {
  for (int r = 0; r < ROWS; r++) {
    for (int c = 0; c < COLS; c++) {
      pixels[r * COLS + c] = IDLE_ICON[r][c] == '#' ? 1 : 0;
    }
  }
  render();
}

void show_idle() {
  busy = false;
  drawIdle();
}

void show_busy() {
  sweep = 0;
  busy = true;
}

// levels: 13 digits, '0' (nothing flagged) to '8' (every reader would flag it)
void show_levels(String levels) {
  busy = false;
  memset(pixels, 0, sizeof(pixels));
  for (int c = 0; c < COLS && c < (int)levels.length(); c++) {
    int level = constrain(levels[c] - '0', 0, ROWS);
    for (int r = ROWS - level; r < ROWS; r++) {
      pixels[r * COLS + c] = 1;
    }
  }
  render();
}

void setup() {
  matrix.begin();
  matrix.clear();
  Bridge.begin();
  Bridge.provide("show_idle", show_idle);
  Bridge.provide("show_busy", show_busy);
  Bridge.provide("show_levels", show_levels);
  drawIdle();
}

void loop() {
  if (busy) {
    memset(pixels, 0, sizeof(pixels));
    for (int r = 0; r < ROWS; r++) {
      pixels[r * COLS + sweep] = 1;
    }
    render();
    sweep = (sweep + 1) % COLS;
    delay(70);
  } else {
    delay(20);
  }
}
