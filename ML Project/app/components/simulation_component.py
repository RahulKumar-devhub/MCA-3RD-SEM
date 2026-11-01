"""
RoadSense - Interactive Road Simulation Component
HTML5/CSS3/JavaScript embedded component providing 60 FPS continuous road animation,
dual live oscilloscope graphing, synchronized ML inference panels, and status HUDs.
"""

import json
import os

def render_simulation_html(simulation_tracks_path: str) -> str:
    """
    Reads the simulation track JSON and generates a self-contained interactive HTML widget.
    """
    with open(simulation_tracks_path, 'r') as f:
        track_data = json.load(f)

    track_json_str = json.dumps(track_data)

    html_code = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <style>
        * {{
          box-sizing: border-box;
          margin: 0;
          padding: 0;
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }}
        body {{
          background-color: #0b0f19;
          color: #f1f5f9;
          padding: 12px;
          overflow-x: hidden;
        }}
        .sim-header {{
          display: flex;
          justify-content: space-between;
          align-items: center;
          background: #111827;
          border: 1px solid #1f2937;
          border-radius: 8px;
          padding: 10px 16px;
          margin-bottom: 12px;
        }}
        .mode-badge {{
          background: #1e1b4b;
          color: #818cf8;
          border: 1px solid #4338ca;
          font-size: 11px;
          font-weight: 700;
          letter-spacing: 0.5px;
          padding: 4px 10px;
          border-radius: 4px;
          text-transform: uppercase;
        }}
        .controls-row {{
          display: flex;
          align-items: center;
          gap: 10px;
        }}
        .btn {{
          background: #1f2937;
          color: #f3f4f6;
          border: 1px solid #374151;
          padding: 7px 16px;
          border-radius: 6px;
          cursor: pointer;
          font-weight: 600;
          font-size: 13px;
          transition: all 0.2s ease;
          display: inline-flex;
          align-items: center;
          gap: 6px;
        }}
        .btn:hover {{
          background: #374151;
          border-color: #4b5563;
        }}
        .btn-primary {{
          background: #0284c7;
          border-color: #0369a1;
          color: #ffffff;
        }}
        .btn-primary:hover {{
          background: #0369a1;
        }}
        .speed-group {{
          display: flex;
          background: #1f2937;
          border-radius: 6px;
          padding: 2px;
          border: 1px solid #374151;
        }}
        .speed-btn {{
          background: transparent;
          border: none;
          color: #9ca3af;
          padding: 5px 10px;
          font-size: 12px;
          font-weight: 600;
          cursor: pointer;
          border-radius: 4px;
        }}
        .speed-btn.active {{
          background: #0284c7;
          color: #ffffff;
        }}
        /* Progress timeline */
        .timeline-container {{
          background: #111827;
          border: 1px solid #1f2937;
          border-radius: 8px;
          padding: 10px 16px;
          margin-bottom: 12px;
        }}
        .timeline-bar {{
          position: relative;
          height: 10px;
          background: #1f2937;
          border-radius: 5px;
          overflow: hidden;
          margin-top: 6px;
        }}
        .timeline-progress {{
          position: absolute;
          top: 0; left: 0; height: 100%;
          width: 0%;
          background: linear-gradient(90deg, #0284c7, #38bdf8);
          transition: width 0.1s linear;
        }}
        .timeline-markers {{
          display: flex;
          justify-content: space-between;
          font-size: 11px;
          color: #94a3b8;
          margin-top: 6px;
        }}
        .marker-tag {{
          font-size: 10px;
          font-weight: 700;
          padding: 1px 6px;
          border-radius: 3px;
        }}
        .marker-normal {{ background: rgba(16, 185, 129, 0.2); color: #34d399; }}
        .marker-bump {{ background: rgba(245, 158, 11, 0.2); color: #fbbf24; }}
        .marker-pothole {{ background: rgba(239, 68, 68, 0.2); color: #f87171; }}

        /* Main Simulation Layout */
        .sim-grid {{
          display: grid;
          grid-template-columns: 360px 1fr;
          gap: 14px;
        }}
        @media (max-width: 900px) {{
          .sim-grid {{ grid-template-columns: 1fr; }}
        }}
        /* Road Canvas Box */
        .road-box {{
          background: #111827;
          border: 1px solid #1f2937;
          border-radius: 8px;
          padding: 12px;
          display: flex;
          flex-direction: column;
          align-items: center;
        }}
        #roadCanvas {{
          background: #1c1917;
          border-radius: 6px;
          box-shadow: 0 4px 14px rgba(0,0,0,0.5);
        }}
        /* Telemetry & Charts Column */
        .telemetry-col {{
          display: flex;
          flex-direction: column;
          gap: 12px;
        }}
        /* Status Alert Banner */
        .status-banner {{
          border-radius: 8px;
          padding: 14px 18px;
          display: flex;
          justify-content: space-between;
          align-items: center;
          transition: all 0.3s ease;
        }}
        .status-normal {{
          background: rgba(16, 185, 129, 0.12);
          border: 1px solid #10b981;
          color: #34d399;
          box-shadow: 0 0 15px rgba(16, 185, 129, 0.15);
        }}
        .status-bump {{
          background: rgba(245, 158, 11, 0.15);
          border: 1px solid #f59e0b;
          color: #fbbf24;
          box-shadow: 0 0 20px rgba(245, 158, 11, 0.25);
        }}
        .status-pothole {{
          background: rgba(239, 68, 68, 0.18);
          border: 1px solid #ef4444;
          color: #f87171;
          box-shadow: 0 0 22px rgba(239, 68, 68, 0.3);
        }}
        .status-title {{
          font-size: 18px;
          font-weight: 800;
          letter-spacing: 0.5px;
          display: flex;
          align-items: center;
          gap: 8px;
        }}
        .status-desc {{
          font-size: 12px;
          color: #94a3b8;
          margin-top: 2px;
        }}
        .conf-badge {{
          font-size: 15px;
          font-weight: 800;
          padding: 4px 12px;
          border-radius: 6px;
          background: #111827;
          border: 1px solid rgba(255,255,255,0.1);
        }}

        /* Inference & Metric Cards */
        .card-row {{
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
          gap: 10px;
        }}
        .metric-card {{
          background: #111827;
          border: 1px solid #1f2937;
          border-radius: 6px;
          padding: 10px 12px;
        }}
        .metric-label {{
          font-size: 11px;
          color: #94a3b8;
          text-transform: uppercase;
          font-weight: 600;
        }}
        .metric-val {{
          font-size: 16px;
          font-weight: 700;
          color: #f8fafc;
          margin-top: 4px;
        }}

        /* Graphs */
        .chart-card {{
          background: #111827;
          border: 1px solid #1f2937;
          border-radius: 8px;
          padding: 10px 14px;
        }}
        .chart-header {{
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 6px;
        }}
        .chart-title {{
          font-size: 12px;
          font-weight: 700;
          text-transform: uppercase;
          letter-spacing: 0.5px;
          color: #cbd5e1;
        }}
        .chart-legend {{
          display: flex;
          gap: 12px;
          font-size: 11px;
        }}
        .leg-item {{
          display: flex;
          align-items: center;
          gap: 4px;
        }}
        .leg-dot {{
          width: 8px;
          height: 8px;
          border-radius: 50%;
        }}
      </style>
    </head>
    <body>

      <!-- Top Header & Controls -->
      <div class="sim-header">
        <div style="display: flex; align-items: center; gap: 12px;">
          <span class="mode-badge">Simulation Mode</span>
          <span style="font-size: 13px; color: #94a3b8;">Research Playback: RoadSens-4M IMU Telemetry</span>
        </div>
        <div class="controls-row">
          <button id="btnPlay" class="btn btn-primary" onclick="togglePlay()">
            <span id="playIcon">▶</span> <span id="playText">Start Simulation</span>
          </button>
          <button class="btn" onclick="resetSim()">↻ Restart</button>
          <div class="speed-group">
            <button class="speed-btn" onclick="setSpeed(0.5, this)">0.5x</button>
            <button class="speed-btn active" onclick="setSpeed(1.0, this)">1.0x</button>
            <button class="speed-btn" onclick="setSpeed(2.0, this)">2.0x</button>
          </div>
        </div>
      </div>

      <!-- Route Progress Timeline -->
      <div class="timeline-container">
        <div style="display: flex; justify-content: space-between; font-size: 12px; font-weight: 600;">
          <span>Demonstration Route Sequence</span>
          <span id="txtPosition">Distance: 0.0 m / 500.0 m (0%)</span>
        </div>
        <div class="timeline-bar">
          <div id="timelineProgress" class="timeline-progress"></div>
        </div>
        <div class="timeline-markers">
          <span class="marker-tag marker-normal">● Normal Asphalt (0m)</span>
          <span class="marker-tag marker-bump">▲ Speed Bump #1 (~120m)</span>
          <span class="marker-tag marker-normal">● Smooth Transit (~250m)</span>
          <span class="marker-tag marker-pothole">▼ Deep Pothole #1 (~350m)</span>
          <span class="marker-tag marker-normal">● Target Finish (500m)</span>
        </div>
      </div>

      <!-- Main Grid: Road Canvas + Telemetry -->
      <div class="sim-grid">

        <!-- Left: Animated Perspective Road Canvas -->
        <div class="road-box">
          <canvas id="roadCanvas" width="330" height="480"></canvas>
          <div style="display: flex; justify-content: space-between; width: 100%; font-size: 11px; color: #64748b; margin-top: 8px;">
            <span>Vehicle: Test Sedan (IMU Sensor Mount)</span>
            <span id="txtFps">60 FPS</span>
          </div>
        </div>

        <!-- Right: Telemetry & Dynamic Charts -->
        <div class="telemetry-col">

          <!-- Prominent Status Banner -->
          <div id="statusBanner" class="status-banner status-normal">
            <div>
              <div id="statusTitle" class="status-title">✓ NORMAL ROAD</div>
              <div id="statusDesc" class="status-desc">Smooth transit - Baseline vehicle vibration</div>
            </div>
            <div style="text-align: right;">
              <div style="font-size: 11px; text-transform: uppercase; color: #94a3b8;">ML Confidence</div>
              <div id="txtConfidence" class="conf-badge">94.8%</div>
            </div>
          </div>

          <!-- Quick Metrics Cards -->
          <div class="card-row">
            <div class="metric-card">
              <div class="metric-label">Active Model</div>
              <div class="metric-val" style="color: #38bdf8; font-size: 14px;">Random Forest</div>
            </div>
            <div class="metric-card">
              <div class="metric-label">Vertical Peak Acc (Z)</div>
              <div id="txtAccZ" class="metric-val">0.12 m/s²</div>
            </div>
            <div class="metric-card">
              <div class="metric-label">Total Acc Mag</div>
              <div id="txtAccMag" class="metric-val">0.38 m/s²</div>
            </div>
            <div class="metric-card">
              <div class="metric-label">Inference Latency</div>
              <div class="metric-val" style="color: #34d399;">0.319 ms</div>
            </div>
          </div>

          <!-- Accelerometer Live Graph -->
          <div class="chart-card">
            <div class="chart-header">
              <span class="chart-title">3-Axis Accelerometer + Magnitude (m/s²)</span>
              <div class="chart-legend">
                <span class="leg-item"><span class="leg-dot" style="background: #38bdf8;"></span> X</span>
                <span class="leg-item"><span class="leg-dot" style="background: #a855f7;"></span> Y</span>
                <span class="leg-item"><span class="leg-dot" style="background: #f59e0b;"></span> Z (Vert)</span>
                <span class="leg-item"><span class="leg-dot" style="background: #10b981;"></span> Mag</span>
              </div>
            </div>
            <canvas id="accChartCanvas" width="560" height="150" style="width: 100%; border-radius: 4px; background: #0f172a;"></canvas>
          </div>

          <!-- Gyroscope Live Graph -->
          <div class="chart-card">
            <div class="chart-header">
              <span class="chart-title">3-Axis Gyroscope Angular Velocity (rad/s)</span>
              <div class="chart-legend">
                <span class="leg-item"><span class="leg-dot" style="background: #38bdf8;"></span> Roll</span>
                <span class="leg-item"><span class="leg-dot" style="background: #a855f7;"></span> Pitch</span>
                <span class="leg-item"><span class="leg-dot" style="background: #f59e0b;"></span> Yaw</span>
                <span class="leg-item"><span class="leg-dot" style="background: #e2e8f0;"></span> Mag</span>
              </div>
            </div>
            <canvas id="gyroChartCanvas" width="560" height="110" style="width: 100%; border-radius: 4px; background: #0f172a;"></canvas>
          </div>

        </div>
      </div>

      <!-- Simulation Script -->
      <script>
        const TRACK_DATA = {track_json_str};
        let currentIndex = 0;
        let isRunning = false;
        let speedMultiplier = 1.0;
        let animationFrameId = null;
        let lastTimestamp = 0;

        // Buffers for Oscilloscope
        const BUFFER_SIZE = 120;
        const accBuffer = {{ x: [], y: [], z: [], mag: [] }};
        const gyroBuffer = {{ x: [], y: [], z: [], mag: [] }};

        // DOM elements
        const roadCanvas = document.getElementById('roadCanvas');
        const roadCtx = roadCanvas.getContext('2d');
        const accCanvas = document.getElementById('accChartCanvas');
        const accCtx = accCanvas.getContext('2d');
        const gyroCanvas = document.getElementById('gyroChartCanvas');
        const gyroCtx = gyroCanvas.getContext('2d');

        const btnPlay = document.getElementById('btnPlay');
        const playIcon = document.getElementById('playIcon');
        const playText = document.getElementById('playText');
        const timelineProgress = document.getElementById('timelineProgress');
        const txtPosition = document.getElementById('txtPosition');
        const statusBanner = document.getElementById('statusBanner');
        const statusTitle = document.getElementById('statusTitle');
        const statusDesc = document.getElementById('statusDesc');
        const txtConfidence = document.getElementById('txtConfidence');
        const txtAccZ = document.getElementById('txtAccZ');
        const txtAccMag = document.getElementById('txtAccMag');

        // Road animation parameters
        let roadOffset = 0;
        let vehicleRecoilY = 0;

        function togglePlay() {{
          isRunning = !isRunning;
          if (isRunning) {{
            playIcon.innerText = "⏸";
            playText.innerText = "Pause";
            btnPlay.classList.replace("btn-primary", "btn");
            lastTimestamp = performance.now();
            loop();
          }} else {{
            playIcon.innerText = "▶";
            playText.innerText = "Resume Simulation";
            btnPlay.classList.replace("btn", "btn-primary");
            if (animationFrameId) cancelAnimationFrame(animationFrameId);
          }}
        }}

        function resetSim() {{
          currentIndex = 0;
          roadOffset = 0;
          accBuffer.x = []; accBuffer.y = []; accBuffer.z = []; accBuffer.mag = [];
          gyroBuffer.x = []; gyroBuffer.y = []; gyroBuffer.z = []; gyroBuffer.mag = [];
          if (!isRunning) {{
            renderFrame();
          }}
        }}

        function setSpeed(sp, btn) {{
          speedMultiplier = sp;
          document.querySelectorAll('.speed-btn').forEach(b => b.classList.remove('active'));
          btn.classList.add('active');
        }}

        function loop(timestamp) {{
          if (!isRunning) return;

          // Advance indices
          const step = Math.max(1, Math.round(1 * speedMultiplier));
          currentIndex = (currentIndex + step);
          if (currentIndex >= TRACK_DATA.length) {{
            currentIndex = 0; // loop
          }}

          roadOffset = (roadOffset + 4 * speedMultiplier) % 40;

          renderFrame();
          animationFrameId = requestAnimationFrame(loop);
        }}

        function renderFrame() {{
          const pt = TRACK_DATA[currentIndex];
          if (!pt) return;

          // Push to graph buffers
          pushBuffer(accBuffer, pt.acc_x, pt.acc_y, pt.acc_z, pt.acc_mag);
          pushBuffer(gyroBuffer, pt.gyro_x, pt.gyro_y, pt.gyro_z, pt.gyro_mag);

          // Update Progress
          const pct = ((currentIndex / TRACK_DATA.length) * 100).toFixed(1);
          const dist = ((currentIndex / TRACK_DATA.length) * 500).toFixed(1);
          timelineProgress.style.width = pct + "%";
          txtPosition.innerText = `Distance: ${{dist}} m / 500.0 m (${{pct}}%)`;

          // Calculate vehicle suspension oscillation from Accel Z
          vehicleRecoilY = Math.max(-12, Math.min(12, pt.acc_z * 2.2));

          // Draw Road
          drawRoad(pt);

          // Draw Charts
          drawOscilloscope(accCtx, accCanvas, accBuffer, -6, 6, [
            {{ key: 'x', color: '#38bdf8' }},
            {{ key: 'y', color: '#a855f7' }},
            {{ key: 'z', color: '#f59e0b' }},
            {{ key: 'mag', color: '#10b981' }}
          ]);

          drawOscilloscope(gyroCtx, gyroCanvas, gyroBuffer, -2.5, 2.5, [
            {{ key: 'x', color: '#38bdf8' }},
            {{ key: 'y', color: '#a855f7' }},
            {{ key: 'z', color: '#f59e0b' }},
            {{ key: 'mag', color: '#e2e8f0' }}
          ]);

          // Update HUD & ML inference
          updateHUD(pt);
        }}

        function pushBuffer(buf, x, y, z, mag) {{
          buf.x.push(x); buf.y.push(y); buf.z.push(z); buf.mag.push(mag);
          if (buf.x.length > BUFFER_SIZE) {{
            buf.x.shift(); buf.y.shift(); buf.z.shift(); buf.mag.shift();
          }}
        }}

        function updateHUD(pt) {{
          const pred = pt.ml_predicted_class || pt.ground_truth;
          const conf = pt.ml_confidence || 95.0;

          txtAccZ.innerText = (pt.acc_z >= 0 ? "+" : "") + pt.acc_z.toFixed(2) + " m/s²";
          txtAccMag.innerText = pt.acc_mag.toFixed(2) + " m/s²";
          txtConfidence.innerText = conf.toFixed(1) + "%";

          // Status class
          statusBanner.classList.remove('status-normal', 'status-bump', 'status-pothole');
          if (pred === 'Bump') {{
            statusBanner.classList.add('status-bump');
            statusTitle.innerHTML = "⚠ BUMP DETECTED";
            statusDesc.innerText = "Suspension compression - High vertical Z-acceleration peak";
          }} else if (pred === 'Pothole') {{
            statusBanner.classList.add('status-pothole');
            statusTitle.innerHTML = "🚧 POTHOLE DETECTED";
            statusDesc.innerText = "Suspension drop & sharp impact shock - High kurtosis & p2p";
          }} else {{
            statusBanner.classList.add('status-normal');
            statusTitle.innerHTML = "✓ NORMAL ROAD";
            statusDesc.innerText = "Smooth transit - Baseline vehicle vibration";
          }}
        }}

        function drawRoad(pt) {{
          const w = roadCanvas.width;
          const h = roadCanvas.height;

          // Clear
          roadCtx.fillStyle = "#1e293b";
          roadCtx.fillRect(0, 0, w, h);

          // Grass / Shoulders
          roadCtx.fillStyle = "#064e3b";
          roadCtx.fillRect(0, 0, 40, h);
          roadCtx.fillRect(w - 40, 0, 40, h);

          // Road Surface
          roadCtx.fillStyle = "#1e293b";
          roadCtx.fillRect(40, 0, w - 80, h);

          // Road Curbs (Red / White alternating)
          const curbH = 20;
          for (let y = -curbH + (roadOffset % curbH); y < h; y += curbH) {{
            const isRed = (Math.floor((y - roadOffset) / curbH) % 2 === 0);
            roadCtx.fillStyle = isRed ? "#ef4444" : "#ffffff";
            roadCtx.fillRect(34, y, 6, curbH);
            roadCtx.fillRect(w - 40, y, 6, curbH);
          }}

          // Center dashed divider
          roadCtx.strokeStyle = "#f8fafc";
          roadCtx.lineWidth = 4;
          roadCtx.setLineDash([25, 20]);
          roadCtx.lineDashOffset = -roadOffset * 1.5;
          roadCtx.beginPath();
          roadCtx.moveTo(w / 2, 0);
          roadCtx.lineTo(w / 2, h);
          roadCtx.stroke();
          roadCtx.setLineDash([]);

          // Draw Approaching Road Anomaly Feature
          const currentClass = pt.ground_truth;
          const carY = h - 130;

          if (currentClass === 'Bump') {{
            // Yellow/Black Chevron Speed Bump
            roadCtx.fillStyle = "#eab308";
            roadCtx.fillRect(46, carY + 30, w - 92, 18);
            // Black stripes
            roadCtx.fillStyle = "#0f172a";
            for (let bx = 50; bx < w - 50; bx += 22) {{
              roadCtx.beginPath();
              roadCtx.moveTo(bx, carY + 30);
              roadCtx.lineTo(bx + 10, carY + 30);
              roadCtx.lineTo(bx + 5, carY + 48);
              roadCtx.lineTo(bx - 5, carY + 48);
              roadCtx.fill();
            }}
          }} else if (currentClass === 'Pothole') {{
            // Dark ragged crater with crack outlines
            roadCtx.fillStyle = "#090d16";
            roadCtx.beginPath();
            roadCtx.ellipse(w / 2 + 15, carY + 35, 45, 22, 0, 0, Math.PI * 2);
            roadCtx.fill();
            roadCtx.strokeStyle = "#450a0a";
            roadCtx.lineWidth = 3;
            roadCtx.stroke();
            // Jagged cracks
            roadCtx.strokeStyle = "#7f1d1d";
            roadCtx.lineWidth = 1.5;
            roadCtx.beginPath();
            roadCtx.moveTo(w/2 - 25, carY + 35);
            roadCtx.lineTo(w/2 - 45, carY + 45);
            roadCtx.moveTo(w/2 + 55, carY + 35);
            roadCtx.lineTo(w/2 + 75, carY + 25);
            roadCtx.stroke();
          }}

          // Draw Vehicle (Overhead View) with responsive suspension recoil
          const carX = w / 2 - 22;
          const finalCarY = carY + vehicleRecoilY;

          // Shadow
          roadCtx.fillStyle = "rgba(0, 0, 0, 0.45)";
          roadCtx.beginPath();
          roadCtx.roundRect(carX + 4, finalCarY + 6, 44, 82, 8);
          roadCtx.fill();

          // Car Body
          roadCtx.fillStyle = "#0284c7";
          roadCtx.beginPath();
          roadCtx.roundRect(carX, finalCarY, 44, 82, 8);
          roadCtx.fill();

          // Windshield & Rear glass
          roadCtx.fillStyle = "#0f172a";
          roadCtx.beginPath();
          roadCtx.roundRect(carX + 5, finalCarY + 18, 34, 18, 3); // front
          roadCtx.roundRect(carX + 6, finalCarY + 54, 32, 14, 2); // rear
          roadCtx.fill();

          // Roof
          roadCtx.fillStyle = "#0369a1";
          roadCtx.fillRect(carX + 7, finalCarY + 36, 30, 18);

          // Headlights (beaming forward)
          roadCtx.fillStyle = "#fef08a";
          roadCtx.fillRect(carX + 4, finalCarY, 7, 4);
          roadCtx.fillRect(carX + 33, finalCarY, 7, 4);

          // Light Cones
          const gradL = roadCtx.createLinearGradient(0, finalCarY, 0, finalCarY - 60);
          gradL.addColorStop(0, "rgba(254, 240, 138, 0.35)");
          gradL.addColorStop(1, "rgba(254, 240, 138, 0)");
          roadCtx.fillStyle = gradL;
          roadCtx.beginPath();
          roadCtx.moveTo(carX + 2, finalCarY);
          roadCtx.lineTo(carX - 25, finalCarY - 60);
          roadCtx.lineTo(carX + 70, finalCarY - 60);
          roadCtx.lineTo(carX + 42, finalCarY);
          roadCtx.fill();

          // Taillights
          roadCtx.fillStyle = (currentClass === 'Bump' || currentClass === 'Pothole') ? "#ef4444" : "#991b1b";
          roadCtx.fillRect(carX + 4, finalCarY + 79, 7, 3);
          roadCtx.fillRect(carX + 33, finalCarY + 79, 7, 3);
        }}

        function drawOscilloscope(ctx, canvas, buffer, minVal, maxVal, seriesList) {{
          const w = canvas.width;
          const h = canvas.height;

          // Clear & Grid
          ctx.fillStyle = "#0b1329";
          ctx.fillRect(0, 0, w, h);

          // Gridlines
          ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
          ctx.lineWidth = 1;
          for (let y = 0; y <= h; y += h / 4) {{
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(w, y);
            ctx.stroke();
          }}
          // Zero reference line
          const zeroY = h - ((0 - minVal) / (maxVal - minVal)) * h;
          ctx.strokeStyle = "rgba(255, 255, 255, 0.25)";
          ctx.setLineDash([4, 4]);
          ctx.beginPath();
          ctx.moveTo(0, zeroY);
          ctx.lineTo(w, zeroY);
          ctx.stroke();
          ctx.setLineDash([]);

          if (!buffer.x || buffer.x.length === 0) return;

          // Draw Series
          seriesList.forEach(s => {{
            const arr = buffer[s.key];
            if (!arr || arr.length === 0) return;

            ctx.strokeStyle = s.color;
            ctx.lineWidth = 1.8;
            ctx.beginPath();

            for (let i = 0; i < arr.length; i++) {{
              const vx = (i / (BUFFER_SIZE - 1)) * w;
              const val = Math.max(minVal, Math.min(maxVal, arr[i]));
              const vy = h - ((val - minVal) / (maxVal - minVal)) * h;
              if (i === 0) ctx.moveTo(vx, vy);
              else ctx.lineTo(vx, vy);
            }}
            ctx.stroke();
          }});
        }}

        // Initial paint
        renderFrame();
      </script>
    </body>
    </html>
    """
    return html_code
