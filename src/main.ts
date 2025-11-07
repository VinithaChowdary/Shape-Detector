import "./style.css";
import { SelectionManager } from "./ui-utils.js";
import { EvaluationManager } from "./evaluation-manager.js";

export interface Point {
  x: number;
  y: number;
}

export interface DetectedShape {
  type: "circle" | "triangle" | "rectangle" | "pentagon" | "star";
  confidence: number;
  boundingBox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  center: Point;
  area: number;
}

export interface DetectionResult {
  shapes: DetectedShape[];
  processingTime: number;
  imageWidth: number;
  imageHeight: number;
}

export class ShapeDetector {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d")!;
  }

  /**
   * MAIN ALGORITHM TO IMPLEMENT
   * Method for detecting shapes in an image
   * @param imageData - ImageData from canvas
   * @returns Promise<DetectionResult> - Detection results
   *
   * TODO: Implement shape detection algorithm here
   */
  async detectShapes(imageData: ImageData): Promise<DetectionResult> {
    const startTime = performance.now();

    // Basic CV pipeline implemented using only browser APIs and math.
    // Steps:
    // 1. Convert to grayscale and binary threshold
    // 2. Find connected components (flood fill)
    // 3. For each component compute bbox, area, centroid, boundary
    // 4. Compute convex hull of boundary and use hull vertex count + circularity/solidity to classify

    const width = imageData.width;
    const height = imageData.height;
    const data = imageData.data;

    // Create binary mask where foreground pixels are 1
    const mask = new Uint8ClampedArray(width * height);

    // Simple threshold: treat nearly-white as background
    for (let i = 0, p = 0; i < data.length; i += 4, p++) {
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
      mask[p] = lum < 250 ? 1 : 0;
    }

    const visited = new Uint8Array(width * height);
    const shapes: DetectedShape[] = [];

    const neighbors = [
      -1,
      1,
      -width,
      width,
      -width - 1,
      -width + 1,
      width - 1,
      width + 1,
    ];

    function idx(x: number, y: number) {
      return y * width + x;
    }

    // Monotone chain convex hull
    function convexHull(points: Point[]): Point[] {
      if (points.length <= 1) return points.slice();
      const pts = points
        .map((p) => ({ x: p.x, y: p.y }))
        .sort((a, b) => (a.x === b.x ? a.y - b.y : a.x - b.x));

      const cross = (o: Point, a: Point, b: Point) =>
        (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);

      const lower: Point[] = [];
      for (const p of pts) {
        while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) {
          lower.pop();
        }
        lower.push(p);
      }

      const upper: Point[] = [];
      for (let i = pts.length - 1; i >= 0; i--) {
        const p = pts[i];
        while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) {
          upper.pop();
        }
        upper.push(p);
      }

      upper.pop();
      lower.pop();
      return lower.concat(upper);
    }

    function polygonArea(points: Point[]): number {
      let a = 0;
      for (let i = 0; i < points.length; i++) {
        const j = (i + 1) % points.length;
        a += points[i].x * points[j].y - points[j].x * points[i].y;
      }
      return Math.abs(a) / 2;
    }

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const id = idx(x, y);
        if (mask[id] && !visited[id]) {
          const stack = [id];
          visited[id] = 1;

          let minX = x,
            maxX = x,
            minY = y,
            maxY = y;
          let area = 0;
          let sumX = 0,
            sumY = 0;

          const componentPoints: Point[] = [];

          while (stack.length) {
            const cur = stack.pop()!;
            const cy = Math.floor(cur / width);
            const cx = cur % width;

            area++;
            sumX += cx;
            sumY += cy;
            componentPoints.push({ x: cx, y: cy });

            if (cx < minX) minX = cx;
            if (cx > maxX) maxX = cx;
            if (cy < minY) minY = cy;
            if (cy > maxY) maxY = cy;

            for (const n of neighbors) {
              const ni = cur + n;
              if (ni < 0 || ni >= width * height) continue;
              if (visited[ni]) continue;
              const ny = Math.floor(ni / width);
              const nx = ni % width;
              if (Math.abs(nx - cx) > 1 || Math.abs(ny - cy) > 1) continue;
              if (mask[ni]) {
                visited[ni] = 1;
                stack.push(ni);
              }
            }
          }

          // ignore tiny noise
          if (area < 20) continue;

          const bbox = {
            x: minX,
            y: minY,
            width: maxX - minX + 1,
            height: maxY - minY + 1,
          };

          const center = { x: sumX / area, y: sumY / area };

          // boundary points
          const boundary: Point[] = [];
          for (const p of componentPoints) {
            const i0 = idx(p.x, p.y);
            let isEdge = false;
            for (const n of neighbors) {
              const ni = i0 + n;
              if (ni < 0 || ni >= width * height) continue;
              const ny = Math.floor(ni / width);
              const nx = ni % width;
              if (Math.abs(nx - p.x) > 1 || Math.abs(ny - p.y) > 1) continue;
              if (!mask[ni]) {
                isEdge = true;
                break;
              }
            }
            if (isEdge) boundary.push(p);
          }

          const perimeter = boundary.length || Math.sqrt(area) * 4;

          const hull = convexHull(boundary.length ? boundary : componentPoints);
          const hullArea = hull.length >= 3 ? polygonArea(hull) : area;
          const hullVertices = hull.length;

          const circularity = perimeter > 0 ? (4 * Math.PI * area) / (perimeter * perimeter) : 0;
          const solidity = hullArea > 0 ? area / hullArea : 0;

          // classification heuristics
          let type: DetectedShape["type"] = "rectangle";
          let confidence = 0.6;

          if (circularity > 0.7 && solidity > 0.8 && hullVertices > 8) {
            type = "circle";
            confidence = 0.7 + Math.min(0.29, (circularity - 0.7));
          } else if (hullVertices === 3) {
            type = "triangle";
            confidence = 0.7 + Math.min(0.29, solidity);
          } else if (hullVertices === 4) {
            type = "rectangle";
            const ar = bbox.width / bbox.height;
            const arScore = 1 - Math.abs(1 - ar);
            confidence = 0.6 + Math.min(0.39, arScore * 0.6 + solidity * 0.4);
          } else if (hullVertices === 5) {
            type = "pentagon";
            confidence = 0.65 + Math.min(0.34, solidity);
          } else {
            if (hullVertices >= 5 && solidity < 0.8 && hullVertices >= 8) {
              type = "star";
              confidence = 0.6 + Math.min(0.39, (0.8 - solidity));
            } else if (hullVertices >= 6 && solidity > 0.9) {
              type = "circle";
              confidence = 0.6 + Math.min(0.39, circularity);
            } else if (hullVertices >= 6) {
              type = solidity < 0.75 ? "star" : "pentagon";
              confidence = 0.55 + Math.min(0.44, solidity);
            } else {
              type = "rectangle";
              confidence = 0.5 + Math.min(0.49, solidity);
            }
          }

          if (!isFinite(confidence) || confidence <= 0) confidence = 0.5;
          if (confidence > 0.99) confidence = 0.99;

          shapes.push({
            type,
            confidence,
            boundingBox: bbox,
            center,
            area,
          });
        }
      }
    }

    const processingTime = performance.now() - startTime;

    return {
      shapes,
      processingTime,
      imageWidth: imageData.width,
      imageHeight: imageData.height,
    };
  }

  loadImage(file: File): Promise<ImageData> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        this.canvas.width = img.width;
        this.canvas.height = img.height;
        this.ctx.drawImage(img, 0, 0);
        const imageData = this.ctx.getImageData(0, 0, img.width, img.height);
        resolve(imageData);
      };
      img.onerror = reject;
      img.src = URL.createObjectURL(file);
    });
  }
}

class ShapeDetectionApp {
  private detector: ShapeDetector;
  private imageInput: HTMLInputElement;
  private resultsDiv: HTMLDivElement;
  private testImagesDiv: HTMLDivElement;
  private evaluateButton: HTMLButtonElement;
  private evaluationResultsDiv: HTMLDivElement;
  private selectionManager: SelectionManager;
  private evaluationManager: EvaluationManager;

  constructor() {
    const canvas = document.getElementById(
      "originalCanvas"
    ) as HTMLCanvasElement;
    this.detector = new ShapeDetector(canvas);

    this.imageInput = document.getElementById("imageInput") as HTMLInputElement;
    this.resultsDiv = document.getElementById("results") as HTMLDivElement;
    this.testImagesDiv = document.getElementById(
      "testImages"
    ) as HTMLDivElement;
    this.evaluateButton = document.getElementById(
      "evaluateButton"
    ) as HTMLButtonElement;
    this.evaluationResultsDiv = document.getElementById(
      "evaluationResults"
    ) as HTMLDivElement;

    this.selectionManager = new SelectionManager();
    this.evaluationManager = new EvaluationManager(
      this.detector,
      this.evaluateButton,
      this.evaluationResultsDiv
    );

    this.setupEventListeners();
    this.loadTestImages().catch(console.error);
  }

  private setupEventListeners(): void {
    this.imageInput.addEventListener("change", async (event) => {
      const file = (event.target as HTMLInputElement).files?.[0];
      if (file) {
        await this.processImage(file);
      }
    });

    this.evaluateButton.addEventListener("click", async () => {
      const selectedImages = this.selectionManager.getSelectedImages();
      await this.evaluationManager.runSelectedEvaluation(selectedImages);
    });
  }

  private async processImage(file: File): Promise<void> {
    try {
      this.resultsDiv.innerHTML = "<p>Processing...</p>";

      const imageData = await this.detector.loadImage(file);
      const results = await this.detector.detectShapes(imageData);

      this.displayResults(results);
    } catch (error) {
      this.resultsDiv.innerHTML = `<p>Error: ${error}</p>`;
    }
  }

  private displayResults(results: DetectionResult): void {
    const { shapes, processingTime } = results;

    let html = `
      <p><strong>Processing Time:</strong> ${processingTime.toFixed(2)}ms</p>
      <p><strong>Shapes Found:</strong> ${shapes.length}</p>
    `;

    if (shapes.length > 0) {
      html += "<h4>Detected Shapes:</h4><ul>";
      shapes.forEach((shape) => {
        html += `
          <li>
            <strong>${
              shape.type.charAt(0).toUpperCase() + shape.type.slice(1)
            }</strong><br>
            Confidence: ${(shape.confidence * 100).toFixed(1)}%<br>
            Center: (${shape.center.x.toFixed(1)}, ${shape.center.y.toFixed(
          1
        )})<br>
            Area: ${shape.area.toFixed(1)}px²
          </li>
        `;
      });
      html += "</ul>";
    } else {
      html +=
        "<p>No shapes detected. Please implement the detection algorithm.</p>";
    }

    this.resultsDiv.innerHTML = html;
  }

  private async loadTestImages(): Promise<void> {
    try {
      const module = await import("./test-images-data.js");
      const testImages = module.testImages;
      const imageNames = module.getAllTestImageNames();

      let html =
        '<h4>Click to upload your own image or use test images for detection. Right-click test images to select/deselect for evaluation:</h4><div class="evaluation-controls"><button id="selectAllBtn">Select All</button><button id="deselectAllBtn">Deselect All</button><span class="selection-info">0 images selected</span></div><div class="test-images-grid">';

      // Add upload functionality as first grid item
      html += `
        <div class="test-image-item upload-item" onclick="triggerFileUpload()">
          <div class="upload-icon">📁</div>
          <div class="upload-text">Upload Image</div>
          <div class="upload-subtext">Click to select file</div>
        </div>
      `;

      imageNames.forEach((imageName) => {
        const dataUrl = testImages[imageName as keyof typeof testImages];
        const displayName = imageName
          .replace(/[_-]/g, " ")
          .replace(/\.(svg|png)$/i, "");
        html += `
          <div class="test-image-item" data-image="${imageName}" 
               onclick="loadTestImage('${imageName}', '${dataUrl}')" 
               oncontextmenu="toggleImageSelection(event, '${imageName}')">
            <img src="${dataUrl}" alt="${imageName}">
            <div>${displayName}</div>
          </div>
        `;
      });

      html += "</div>";
      this.testImagesDiv.innerHTML = html;

      this.selectionManager.setupSelectionControls();

      (window as any).loadTestImage = async (name: string, dataUrl: string) => {
        try {
          const response = await fetch(dataUrl);
          const blob = await response.blob();
          const file = new File([blob], name, { type: "image/svg+xml" });

          const imageData = await this.detector.loadImage(file);
          const results = await this.detector.detectShapes(imageData);
          this.displayResults(results);

          console.log(`Loaded test image: ${name}`);
        } catch (error) {
          console.error("Error loading test image:", error);
        }
      };

      (window as any).toggleImageSelection = (
        event: MouseEvent,
        imageName: string
      ) => {
        event.preventDefault();
        this.selectionManager.toggleImageSelection(imageName);
      };

      // Add upload functionality
      (window as any).triggerFileUpload = () => {
        this.imageInput.click();
      };
    } catch (error) {
      this.testImagesDiv.innerHTML = `
        <p>Test images not available. Run 'node convert-svg-to-png.js' to generate test image data.</p>
        <p>SVG files are available in the test-images/ directory.</p>
      `;
    }
  }
}

document.addEventListener("DOMContentLoaded", () => {
  new ShapeDetectionApp();
});
