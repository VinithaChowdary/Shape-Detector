# Shape Detection Challenge



# Shape Detector

A small TypeScript web app that detects and classifies simple geometric shapes (circle, triangle, rectangle, pentagon, star) in images. The UI lets you upload images or use provided test SVGs and shows detected shapes with basic metadata.

## Quick start

Prerequisites:

- Node.js 16+ and npm (or yarn)

Install dependencies and start the dev server:

```bash
cd /path/to/shape-detector
npm install
npm run dev
```

Open http://localhost:5173 (or the port printed by Vite) in your browser.

## Project structure

Key files and folders:

```
shape-detector/
├── src/                   # TypeScript source
│   ├── main.ts            # App entry + ShapeDetector implementation
│   ├── evaluation.ts      # Evaluation helper / runner (optional)
│   └── ...                # other UI and util modules
├── test-images/           # Inline SVG test images used by the UI
├── ground_truth.json      # Ground-truth data used by the optional evaluation runner
├── index.html             # Application UI
└── README.md              # This file
```

## Usage

- Launch the dev server (`npm run dev`) and use the web UI to upload images.
- Click a test image in the gallery to run detection on that built-in example.
- For each detected shape the UI reports: type, confidence, center, and area.

Programmatic usage:

The main detector is implemented as the `ShapeDetector` class in `src/main.ts`. It exposes:

- `loadImage(file: File): Promise<ImageData>` — load an image file into a canvas and return its ImageData.
- `detectShapes(imageData: ImageData): Promise<DetectionResult>` — run detection and return a result object containing detected shapes and timing.

## Development notes

- The detection algorithm is implemented using pure browser APIs (Canvas ImageData) and geometric heuristics — no external CV libraries.
- If you need to tune detection sensitivity (thresholds, area filtering, vertex heuristics), edit `src/main.ts` where helper heuristics are documented as comments.
- To build for production:

```bash
npm run build
```

## License & attribution

This project is provided as-is. Feel free to adapt the code for learning or demos.

---

If you want, I can:
- Run the app locally and show evaluation results against `ground_truth.json`.
- Tweak the README further (add screenshots, badges, or contributor info).
Let me know which you'd prefer.
- Correctly identifying all shapes present in test images

- Minimizing false positives (detecting shapes that aren't there)

