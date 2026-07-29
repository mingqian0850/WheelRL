"""Local browser controls for the interactive whole-body controller."""

from __future__ import annotations

import json
import queue
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

PANEL_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>WheelRL TCP control</title>
<style>
:root { color-scheme: dark; font-family: Inter, system-ui, sans-serif; }
body { margin: 0; background: #101318; color: #e8edf2; }
main { max-width: 880px; margin: auto; padding: 18px; }
h1 { margin: 0 0 4px; font-size: 1.45rem; }
.sub { color: #9ba9b8; margin-bottom: 16px; }
.cards { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
.card, section { background: #191e26; border: 1px solid #2b3440;
  border-radius: 10px; padding: 12px; }
.label { color: #9ba9b8; font-size: .78rem; text-transform: uppercase; }
.value { font: 600 1.05rem ui-monospace, monospace; margin-top: 5px; }
section { margin-top: 10px; }
h2 { font-size: 1rem; margin: 0 0 10px; }
.pose { display: grid; grid-template-columns: repeat(6, 1fr); gap: 8px; }
label { color: #aeb8c4; font-size: .8rem; }
input[type=number] { box-sizing: border-box; width: 100%; margin-top: 4px;
  padding: 8px; color: #fff; background: #0e1217; border: 1px solid #3a4553;
  border-radius: 6px; font: .95rem ui-monospace, monospace; }
.jog { display: grid; grid-template-columns: repeat(6, 1fr); gap: 7px;
  margin-top: 9px; }
button { padding: 9px 8px; color: #e8edf2; background: #283240;
  border: 1px solid #3b4858; border-radius: 6px; cursor: pointer; }
button:hover { background: #344154; }
button.primary { background: #2563eb; border-color: #3b82f6; }
button.good { background: #12613b; border-color: #1a8050; }
button.warn { background: #7a3d15; border-color: #a8561e; }
.actions { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
.toggle { margin-left: auto; display: flex; align-items: center; gap: 7px; }
.hint { color: #98a6b5; font-size: .82rem; margin-top: 9px; }
#status.bad { color: #ff7b72; }
@media (max-width: 700px) {
  .pose, .jog { grid-template-columns: repeat(3, 1fr); }
  .cards { grid-template-columns: 1fr; }
}
</style>
</head>
<body><main>
<h1>B2-W + Z1 TCP controller</h1>
<div class="sub">The MuJoCo window remains dedicated to camera and viewer controls.</div>
<div class="cards">
  <div class="card"><div class="label">Connection</div>
    <div id="status" class="value">Connecting…</div></div>
  <div class="card"><div class="label">Tracking error</div>
    <div id="error" class="value">—</div></div>
  <div class="card"><div class="label">Base / gripper</div>
    <div id="mode" class="value">—</div></div>
</div>

<section>
  <h2>Target offset from startup home</h2>
  <div class="pose">
    <label>X · forward (m)<input id="x" type="number" step="0.01"></label>
    <label>Y · left (m)<input id="y" type="number" step="0.01"></label>
    <label>Z · up (m)<input id="z" type="number" step="0.01"></label>
    <label>Roll (deg)<input id="roll" type="number" step="2"></label>
    <label>Pitch (deg)<input id="pitch" type="number" step="2"></label>
    <label>Yaw (deg)<input id="yaw" type="number" step="2"></label>
  </div>
  <div class="jog">
    <button onclick="jog('x',.01)">Forward +X</button>
    <button onclick="jog('x',-.01)">Back −X</button>
    <button onclick="jog('y',.01)">Left +Y</button>
    <button onclick="jog('y',-.01)">Right −Y</button>
    <button onclick="jog('z',.01)">Up +Z</button>
    <button onclick="jog('z',-.01)">Down −Z</button>
    <button onclick="jog('roll',2)">Roll +</button>
    <button onclick="jog('roll',-2)">Roll −</button>
    <button onclick="jog('pitch',2)">Pitch +</button>
    <button onclick="jog('pitch',-2)">Pitch −</button>
    <button onclick="jog('yaw',2)">Yaw +</button>
    <button onclick="jog('yaw',-2)">Yaw −</button>
  </div>
  <div class="actions" style="margin-top:10px">
    <button class="primary" onclick="sendPose()">Apply TCP pose</button>
    <button onclick="home()">Home</button>
    <button onclick="useTarget()">Reload current target</button>
  </div>
  <div class="hint">Offsets use the coordinate axes captured at startup.
    A target outside the arm workspace activates the wheeled base automatically.</div>
</section>

<section>
  <h2>Gripper and run controls</h2>
  <div class="actions">
    <button class="good" onclick="command({action:'gripper',closed:true})">
      Close gripper</button>
    <button class="warn" onclick="command({action:'gripper',closed:false})">
      Open gripper</button>
    <button onclick="command({action:'print_pose'})">Print pose in terminal</button>
    <button onclick="command({action:'quit'})">End simulation</button>
    <label class="toggle"><input id="auto" type="checkbox"
      onchange="command({action:'auto_drive',enabled:this.checked})">
      Automatic base motion</label>
  </div>
</section>
</main>
<script>
const ids = ['x','y','z','roll','pitch','yaw'];
let latest = null, initialized = false;
async function command(payload) {
  const response = await fetch('/api/command', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body:JSON.stringify(payload)
  });
  if (!response.ok) throw new Error(await response.text());
}
function values() { return ids.map(id => Number(document.getElementById(id).value)); }
function sendPose() {
  const v = values();
  command({action:'set_pose',position_offset:v.slice(0,3),
    rpy_offset_deg:v.slice(3)});
}
function jog(id, delta) {
  const input = document.getElementById(id);
  input.value = (Number(input.value || 0) + delta).toFixed(id.length === 1 ? 3 : 1);
  sendPose();
}
function populate(state) {
  const values = [...state.position_offset, ...state.rpy_offset_deg];
  ids.forEach((id, i) => document.getElementById(id).value =
    values[i].toFixed(i < 3 ? 3 : 1));
}
function useTarget() { if (latest) populate(latest); }
function home() {
  ids.forEach(id => document.getElementById(id).value = '0');
  command({action:'home'});
}
async function poll() {
  try {
    const response = await fetch('/api/state', {cache:'no-store'});
    if (!response.ok) throw new Error('HTTP ' + response.status);
    latest = await response.json();
    if (!initialized) { populate(latest); initialized = true; }
    document.getElementById('status').textContent = 'Connected · running';
    document.getElementById('status').className = 'value';
    document.getElementById('error').textContent =
      latest.position_error.toFixed(4) + ' m · ' +
      latest.orientation_error_deg.toFixed(2) + '°';
    document.getElementById('mode').textContent =
      (latest.mobile_base_active ? 'base moving' : 'arm workspace') + ' · ' +
      (latest.gripper_closed ? 'closed' : 'open');
    document.getElementById('auto').checked = latest.auto_drive;
  } catch (error) {
    document.getElementById('status').textContent = 'Disconnected';
    document.getElementById('status').className = 'value bad';
  }
}
poll(); setInterval(poll, 150);
</script></body></html>
"""


class WBCControlPanel:
    """Threaded localhost server that exchanges JSON commands with the sim."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        self._host = host
        self._port = port
        self._commands: queue.SimpleQueue[dict[str, Any]] = queue.SimpleQueue()
        self._state: dict[str, Any] = {
            "position_offset": [0.0, 0.0, 0.0],
            "rpy_offset_deg": [0.0, 0.0, 0.0],
            "position_error": 0.0,
            "orientation_error_deg": 0.0,
            "mobile_base_active": False,
            "gripper_closed": False,
            "auto_drive": True,
        }
        self._state_lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        if self._server is None:
            raise RuntimeError("control panel has not started")
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/"

    def start(self) -> None:
        panel = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:
                return

            def _send(
                self,
                status: HTTPStatus,
                body: bytes,
                content_type: str,
            ) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                if self.path == "/":
                    self._send(
                        HTTPStatus.OK,
                        PANEL_HTML.encode(),
                        "text/html; charset=utf-8",
                    )
                elif self.path == "/api/state":
                    with panel._state_lock:
                        body = json.dumps(panel._state).encode()
                    self._send(HTTPStatus.OK, body, "application/json")
                else:
                    self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")

            def do_POST(self) -> None:
                if self.path != "/api/command":
                    self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 16_384:
                        raise ValueError("invalid content length")
                    command = json.loads(self.rfile.read(length))
                    if not isinstance(command, dict) or "action" not in command:
                        raise ValueError("command must contain action")
                    panel._commands.put(command)
                except (ValueError, json.JSONDecodeError) as error:
                    self._send(
                        HTTPStatus.BAD_REQUEST,
                        str(error).encode(),
                        "text/plain; charset=utf-8",
                    )
                    return
                self._send(HTTPStatus.NO_CONTENT, b"", "application/json")

        self._server = ThreadingHTTPServer((self._host, self._port), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="wheelrl-control-panel",
            daemon=True,
        )
        self._thread.start()

    def open_browser(self) -> bool:
        try:
            return webbrowser.open(self.url, new=1)
        except webbrowser.Error:
            return False

    def publish(self, state: dict[str, Any]) -> None:
        with self._state_lock:
            self._state = state

    def drain_commands(self) -> list[dict[str, Any]]:
        commands: list[dict[str, Any]] = []
        while not self._commands.empty():
            commands.append(self._commands.get_nowait())
        return commands

    def close(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._server = None
        self._thread = None
