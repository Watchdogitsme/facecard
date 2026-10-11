"""
Browser-Based Web Mirror for Face Skin Evaluation & Celebrity Matcher
Runs a local lightweight HTTP server with zero extra dependencies (pure Python + OpenCV).
Allows users to use their laptop camera directly via Chrome/Edge/Firefox,
making it 100% compatible with Windows, Mac, and WSL environments!
"""

import json
import base64
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler

import cv2
import numpy as np

from skin_analyzer import SkinAnalyzer
from age_estimator import AgeEstimator
from celebrity_matcher import CelebrityMatcher
from face_detector import RobustFaceDetector

# Global engine instances
skin_analyzer = SkinAnalyzer()
age_estimator = AgeEstimator()
celeb_matcher = CelebrityMatcher(target_probability=7.1)
face_detector = RobustFaceDetector()


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Biometric & Skin Mirror</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  body { background: #0f1217; color: #f0f2f5; min-height: 100vh; padding: 20px; display: flex; flex-direction: column; align-items: center; }
  header { text-align: center; margin-bottom: 20px; }
  header h1 { font-size: 26px; color: #f5c518; letter-spacing: 1px; display: flex; align-items: center; justify-content: center; gap: 10px; }
  header p { font-size: 14px; color: #9aa0a6; margin-top: 4px; }
  
  .container { display: flex; flex-wrap: wrap; gap: 24px; max-width: 1200px; width: 100%; justify-content: center; }
  
  .video-panel { flex: 1; min-width: 340px; max-width: 640px; background: #161a22; border-radius: 12px; border: 1px solid #282e3d; overflow: hidden; display: flex; flex-direction: column; position: relative; }
  .video-container { position: relative; width: 100%; aspect-ratio: 4/3; background: #000; overflow: hidden; display: flex; align-items: center; justify-content: center; }
  video, canvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover; }
  #overlayCanvas { pointer-events: none; }
  
  .controls { display: flex; gap: 12px; padding: 14px; background: #1a1f29; border-top: 1px solid #282e3d; align-items: center; justify-content: space-between; flex-wrap: wrap; }
  button { background: #2563eb; color: #fff; border: none; border-radius: 6px; padding: 9px 18px; font-weight: 600; cursor: pointer; transition: all 0.2s; font-size: 13px; }
  button:hover { background: #1d4ed8; }
  button.secondary { background: #374151; }
  button.secondary:hover { background: #4b5563; }
  
  .dashboard { flex: 1; min-width: 340px; max-width: 440px; display: flex; flex-direction: column; gap: 16px; }
  
  .card { background: #161a22; border: 1px solid #282e3d; border-radius: 12px; padding: 18px; box-shadow: 0 4px 12px rgba(0,0,0,0.3); }
  .card-title { font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 12px; display: flex; align-items: center; gap: 8px; }
  
  /* Celebrity Card */
  .celeb-card .card-title { color: #60a5fa; }
  .celeb-body { display: flex; gap: 16px; align-items: center; }
  .celeb-img-frame { width: 110px; height: 110px; border-radius: 10px; overflow: hidden; border: 2px solid #3b82f6; flex-shrink: 0; background: #1f2937; }
  .celeb-img-frame img { width: 100%; height: 100%; object-fit: cover; }
  .celeb-info { flex: 1; }
  .celeb-name { font-size: 18px; font-weight: bold; color: #fff; margin-bottom: 2px; }
  .celeb-title { font-size: 12px; color: #9ca3af; margin-bottom: 10px; }
  .prob-badge { display: flex; align-items: baseline; gap: 6px; margin-bottom: 6px; }
  .prob-val { font-size: 20px; font-weight: 800; color: #10b981; }
  .prob-label { font-size: 12px; color: #6ee7b7; }
  
  /* Skin Card */
  .skin-card .card-title { color: #f472b6; }
  .rating-badge { display: flex; align-items: center; gap: 14px; margin-bottom: 14px; }
  .big-score { font-size: 34px; font-weight: 900; color: #fff; background: #1f242e; border: 2px solid #10b981; padding: 4px 14px; border-radius: 10px; }
  .status-text { font-size: 15px; font-weight: 600; color: #10b981; }
  
  .progress-group { margin-top: 8px; }
  .progress-row { display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 4px; color: #d1d5db; }
  .bar-bg { width: 100%; height: 8px; background: #282e3d; border-radius: 4px; overflow: hidden; margin-bottom: 10px; }
  .bar-fill { height: 100%; border-radius: 4px; transition: width 0.3s ease; }
  .bar-smooth { background: #38bdf8; }
  .bar-clarity { background: #fb923c; }
  .bar-tone { background: #4ade80; }
  .bar-match { background: #10b981; }
  
  /* Age Card */
  .age-card .card-title { color: #fbbf24; }
  .age-body { display: flex; align-items: center; gap: 16px; }
  .age-val-box { font-size: 32px; font-weight: 900; color: #fff; background: #1f242e; border: 2px solid #fbbf24; padding: 4px 14px; border-radius: 10px; }
  .age-details { font-size: 13px; line-height: 1.5; color: #cbd5e1; }
  
  .badge-tag { display: inline-block; padding: 2px 8px; background: #1e293b; border-radius: 4px; font-size: 11px; color: #94a3b8; }
  .status-pill { font-size: 12px; padding: 4px 10px; border-radius: 20px; background: rgba(16, 185, 129, 0.15); color: #34d399; display: inline-flex; align-items: center; gap: 6px; }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: #10b981; }
</style>
</head>
<body>
<header>
  <h1><span>★</span> AI BIOMETRIC & SKIN EVALUATION MIRROR <span>★</span></h1>
  <p>Real-time Computer Vision Face Detection, Skin Analysis, Age Estimation & Celebrity Matching</p>
</header>

<div class="container">
  <!-- Webcam Video Feed -->
  <div class="video-panel">
    <div class="video-container">
      <video id="webcam" autoplay playsinline muted></video>
      <canvas id="overlayCanvas"></canvas>
    </div>
    <div class="controls">
      <div class="status-pill" id="liveStatus">
        <div class="dot"></div>
        <span id="statusLabel">Camera Active</span>
      </div>
      <div style="display: flex; gap: 8px;">
        <button id="cycleBtn" class="secondary" onclick="cycleCelebrity()">Cycle Celebrity</button>
        <button id="snapBtn" onclick="takeSnapshot()">Take Snapshot</button>
      </div>
    </div>
  </div>

  <!-- Right Biometric Dashboard -->
  <div class="dashboard">
    <!-- Celebrity Match Card -->
    <div class="card celeb-card">
      <div class="card-title" style="display: flex; justify-content: space-between; align-items: center;">
        <span>★ #1 Top Look-Alike (12 Checks)</span>
        <span class="badge-tag" id="celebCat" style="color: #60a5fa; border: 1px solid #3b82f6;">YOUTUBER</span>
      </div>
      <div class="celeb-body">
        <div class="celeb-img-frame">
          <img id="celebImg" src="/celebrity_card?id=mrbeast" alt="Top Look-Alike Match">
        </div>
        <div class="celeb-info">
          <div class="celeb-name" id="celebName">MrBeast</div>
          <div class="celeb-title" id="celebTitle">YouTube Icon & Philanthropist</div>
          <div class="prob-badge">
            <span class="prob-val" id="celebProb">7.2 / 10</span>
            <span class="prob-label" id="celebPercent">(72% Match)</span>
          </div>
          <div class="bar-bg">
            <div class="bar-fill bar-match" id="celebBar" style="width: 72%;"></div>
          </div>
          <div id="celebTraits" style="font-size: 11px; color: #93c5fd; margin-top: 4px;">Top Traits: Face Ratio (94%), Eye Spacing (88%)</div>
        </div>
      </div>
    </div>

    <!-- Skin Evaluation Card -->
    <div class="card skin-card">
      <div class="card-title">♥ Skin Quality Rating</div>
      <div class="rating-badge">
        <div class="big-score" id="skinScore">8.4</div>
        <div>
          <div style="font-size: 11px; color: #9ca3af;">Complexion Grade:</div>
          <div class="status-text" id="skinGrade">Smooth & Radiant</div>
        </div>
      </div>
      <div class="progress-group">
        <div class="progress-row"><span>Smoothness</span><span id="valSmooth">8.2 / 10</span></div>
        <div class="bar-bg"><div class="bar-fill bar-smooth" id="barSmooth" style="width: 82%;"></div></div>

        <div class="progress-row"><span>Even Tone</span><span id="valTone">8.6 / 10</span></div>
        <div class="bar-bg"><div class="bar-fill bar-tone" id="barTone" style="width: 86%;"></div></div>

        <div class="progress-row"><span>Blemish Clarity</span><span id="valClarity">8.5 / 10</span></div>
        <div class="bar-bg"><div class="bar-fill bar-clarity" id="barClarity" style="width: 85%;"></div></div>
      </div>
    </div>

    <!-- Age Estimation Card -->
    <div class="card age-card">
      <div class="card-title">⌚ Facial Age Estimation</div>
      <div class="age-body">
        <div class="age-val-box" id="ageVal">~24</div>
        <div class="age-details">
          <div><strong>Bracket:</strong> <span id="ageRange">22 - 26 yrs</span></div>
          <div><strong>Texture:</strong> <span id="ageDesc">Youthful & Fresh</span></div>
          <div style="margin-top: 4px;"><span class="badge-tag">Wrinkle Level: Minimal</span></div>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
  const video = document.getElementById('webcam');
  const canvas = document.getElementById('overlayCanvas');
  const ctx = canvas.getContext('2d');
  let isAnalyzing = false;

  async function initCamera() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
      video.srcObject = stream;
      video.onloadedmetadata = () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        startProcessingLoop();
      };
    } catch (err) {
      console.error("Camera access error:", err);
      document.getElementById('statusLabel').textContent = "Camera access denied / not found";
    }
  }

  function startProcessingLoop() {
    setInterval(async () => {
      if (isAnalyzing || video.paused || video.ended) return;
      isAnalyzing = true;
      try {
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = 320;
        tempCanvas.height = 240;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.drawImage(video, 0, 0, 320, 240);
        const dataUrl = tempCanvas.toDataURL('image/jpeg', 0.65);

        const res = await fetch('/analyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image: dataUrl })
        });
        const data = await res.json();
        updateUI(data);
      } catch (e) {
        console.error("Analysis error:", e);
      } finally {
        isAnalyzing = false;
      }
    }, 200); // 5 times per sec
  }

  function updateUI(data) {
    if (!data) return;
    // Skin
    if (data.skin) {
      document.getElementById('skinScore').textContent = data.skin.overall_score.toFixed(1);
      document.getElementById('skinGrade').textContent = data.skin.status;
      document.getElementById('valSmooth').textContent = data.skin.smoothness.toFixed(1) + " / 10";
      document.getElementById('barSmooth').style.width = (data.skin.smoothness * 10) + "%";
      document.getElementById('valTone').textContent = data.skin.uniformity.toFixed(1) + " / 10";
      document.getElementById('barTone').style.width = (data.skin.uniformity * 10) + "%";
      document.getElementById('valClarity').textContent = data.skin.clarity.toFixed(1) + " / 10";
      document.getElementById('barClarity').style.width = (data.skin.clarity * 10) + "%";
    }

    // Age
    if (data.age) {
      document.getElementById('ageVal').textContent = "~" + data.age.estimated_age;
      document.getElementById('ageRange').textContent = data.age.age_range;
      document.getElementById('ageDesc').textContent = data.age.descriptor;
    }

    // Celebrity
    if (data.celebrity) {
      document.getElementById('celebName').textContent = data.celebrity.name;
      document.getElementById('celebTitle').textContent = data.celebrity.title;
      if (data.celebrity.category) document.getElementById('celebCat').textContent = data.celebrity.category;
      document.getElementById('celebProb').textContent = data.celebrity.probability.toFixed(1) + " / 10";
      document.getElementById('celebPercent').textContent = "(" + Math.round(data.celebrity.probability * 10) + "% Match)";
      document.getElementById('celebBar').style.width = (data.celebrity.probability * 10) + "%";
      document.getElementById('celebImg').src = "/celebrity_card?id=" + data.celebrity.id;
      if (data.celebrity.top_traits && data.celebrity.top_traits.length > 0) {
        const traitsStr = data.celebrity.top_traits.map(t => `${t[0]} (${t[1]}%)`).join(', ');
        document.getElementById('celebTraits').textContent = "Top Traits: " + traitsStr;
      }
    }

    // Face Overlay
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (data.face_box) {
      const [fx, fy, fw, fh] = data.face_box;
      // Scale from 320x240 to canvas dimensions
      const sx = canvas.width / 320.0;
      const sy = canvas.height / 240.0;
      const bx = fx * sx, by = fy * sy, bw = fw * sx, bh = fh * sy;

      ctx.strokeStyle = '#00ffc8';
      ctx.lineWidth = 3;
      ctx.strokeRect(bx, by, bw, bh);

      ctx.fillStyle = 'rgba(0, 255, 200, 0.2)';
      ctx.fillRect(bx, by, bw, bh);
    }
  }

  async function cycleCelebrity() {
    await fetch('/cycle_celeb', { method: 'POST' });
  }

  function takeSnapshot() {
    window.print();
  }

  window.addEventListener('load', initCamera);
</script>
</body>
</html>
"""


class WebMirrorHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))

        elif self.path.startswith("/celebrity_card"):
            # Return celebrity card image as JPEG
            celeb_id = "leonardo_dicaprio"
            if "id=" in self.path:
                celeb_id = self.path.split("id=")[-1].split("&")[0]

            celeb_id_norm = celeb_id.lower().strip()
            celeb_entry = next((c for c in celeb_matcher.celebrities if c['id'].lower().strip() == celeb_id_norm), celeb_matcher.celebrities[0])
            card_img = celeb_matcher.get_celebrity_image(celeb_entry)
            ret, buf = cv2.imencode(".jpg", card_img)
            if ret:
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.end_headers()
                self.wfile.write(buf.tobytes())
            else:
                self.send_error(500)
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/analyze":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)
            try:
                payload = json.loads(post_data.decode("utf-8"))
                img_data = payload.get("image", "")
                if "," in img_data:
                    img_data = img_data.split(",", 1)[1]

                img_bytes = base64.b64decode(img_data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is None:
                    self._send_json({"error": "Frame decode error"})
                    return

                # Detect Face with Deep Learning YuNet & Landmarks
                detailed_faces = face_detector.detect_faces_detailed(frame, scale_factor=1.15, min_neighbors=4, min_size=(50, 50))

                if len(detailed_faces) > 0:
                    largest = max(detailed_faces, key=lambda d: d["box"][2] * d["box"][3])
                    fx, fy, fw, fh = largest["box"]
                    landmarks = largest.get("landmarks")
                    face_crop = frame[fy:fy + fh, fx:fx + fw]

                    crop_landmarks = None
                    if landmarks is not None:
                        crop_landmarks = landmarks.copy()
                        crop_landmarks[:, 0] -= fx
                        crop_landmarks[:, 1] -= fy

                    skin = skin_analyzer.evaluate_face(face_crop, landmarks=crop_landmarks)
                    age = age_estimator.estimate(face_crop)
                    celeb_dict, celeb_prob, _, top_traits = celeb_matcher.match(face_crop, landmarks=crop_landmarks)

                    clean_skin = {
                        'overall_score': skin['overall_score'],
                        'smoothness': skin['smoothness'],
                        'clarity': skin['clarity'],
                        'uniformity': skin['uniformity'],
                        'status': skin['status']
                    }
                    clean_celeb = {
                        'id': celeb_dict['id'],
                        'name': celeb_dict['name'],
                        'title': celeb_dict['title'],
                        'category': celeb_dict.get('category', 'STAR'),
                        'probability': celeb_prob,
                        'top_traits': top_traits
                    }

                    self._send_json({
                        'face_box': [fx, fy, fw, fh],
                        'skin': clean_skin,
                        'age': age,
                        'celebrity': clean_celeb
                    })
                else:
                    self._send_json({'face_box': None})

            except Exception as e:
                self._send_json({"error": str(e)})

        elif self.path == "/cycle_celeb":
            celeb_matcher.cycle_next()
            self._send_json({"status": "cycled"})
        else:
            self.send_error(404)

    def _send_json(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def log_message(self, format, *args):
        # Suppress routine HTTP log messages for smooth console output
        return


def run_server(port=5000):
    server_address = ("", port)
    httpd = HTTPServer(server_address, WebMirrorHandler)
    print(f"\n=========================================================")
    print(f"★ AI Biometric Mirror Web App running at:")
    print(f"  👉 http://localhost:{port}")
    print(f"=========================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping web server.")
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000, help="Web server port (default: 5000)")
    args = parser.parse_args()
    run_server(args.port)
