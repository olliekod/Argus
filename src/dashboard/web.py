"""
Argus Local Web Dashboard
=========================

Lightweight localhost-only dashboard for debugging and command access.
Runs alongside the headless main process, reading status via callbacks.
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from aiohttp import web

logger = logging.getLogger('argus.dashboard')

# Inline HTML template (single page, no external deps)
_HTML = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Argus Dashboard</title>
<meta http-equiv="refresh" content="30">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Consolas','Courier New',monospace;background:#1a1a2e;color:#e0e0e0;padding:16px}
h1{color:#00d4ff;margin-bottom:12px;font-size:1.4em}
h2{color:#00d4ff;margin:16px 0 8px;font-size:1.1em;border-bottom:1px solid #333;padding-bottom:4px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.card{background:#16213e;border:1px solid #0f3460;border-radius:8px;padding:12px}
.ok{color:#4caf50}.warn{color:#ff9800}.err{color:#f44336}
table{width:100%;border-collapse:collapse;font-size:0.85em}
td,th{text-align:left;padding:4px 8px;border-bottom:1px solid #222}
th{color:#888}
pre{background:#0d1117;padding:8px;border-radius:4px;max-height:300px;overflow-y:auto;font-size:0.8em;white-space:pre-wrap}
input[type=text]{background:#0d1117;color:#e0e0e0;border:1px solid #0f3460;padding:6px 10px;border-radius:4px;width:70%;font-family:monospace}
button{background:#0f3460;color:#e0e0e0;border:none;padding:6px 16px;border-radius:4px;cursor:pointer;margin-left:8px}
button:hover{background:#1a4a8a}
#cmd-output{margin-top:8px}
.stat{font-size:1.4em;font-weight:bold}
.label{color:#888;font-size:0.85em}
</style></head><body>
<h1>Argus Dashboard</h1>
<div class="grid">
<div class="card">
<h2>System</h2>
<div id="system">Loading...</div>
</div>
<div class="card">
<h2>Providers</h2>
<div id="providers">Loading...</div>
</div>
<div class="card">
<h2>P&amp;L Summary</h2>
<div id="pnl">Loading...</div>
</div>
<div class="card">
<h2>Farm</h2>
<div id="farm">Loading...</div>
</div>
</div>
<div class="card" style="margin-top:16px">
<h2>Command</h2>
<div>
<input type="text" id="cmd-input" placeholder="/pnl, /status, /positions, /zombies, /dashboard ..." onkeydown="if(event.key==='Enter')runCmd()">
<button onclick="runCmd()">Run</button>
</div>
<pre id="cmd-output"></pre>
</div>
<div class="card" style="margin-top:16px">
<h2>Recent Logs</h2>
<pre id="logs">Loading...</pre>
</div>
<script>
async function load(){
  try{
    let r=await fetch('/api/status');let d=await r.json();
    // System
    let s=d.system||{};
    document.getElementById('system').innerHTML=
      '<div><span class="label">Uptime:</span> '+s.uptime+'</div>'+
      '<div><span class="label">DB Size:</span> '+(s.db_size_mb||'?')+' MB</div>'+
      '<div><span class="label">Boot phases:</span></div>'+
      '<pre>'+(s.boot_phases||'N/A')+'</pre>';
    // Providers
    let p=d.providers||{};let ph='<table><tr><th>Provider</th><th>Status</th><th>Last Msg</th><th>Reconnects</th></tr>';
    for(let k in p){let v=p[k];let cls=v.connected?'ok':'err';
      ph+='<tr><td>'+k+'</td><td class="'+cls+'">'+(v.connected?'Connected':'Disconnected')+'</td><td>'+(v.seconds_since_last_message||'-')+'s</td><td>'+(v.reconnect_attempts||0)+'</td></tr>';}
    ph+='</table>';document.getElementById('providers').innerHTML=ph;
    // PnL
    let pnl=d.pnl||{};
    document.getElementById('pnl').innerHTML=
      '<div><span class="label">Today:</span> <span class="stat '+(pnl.today_pnl>=0?'ok':'err')+'">$'+(pnl.today_pnl||0).toFixed(2)+'</span></div>'+
      '<div><span class="label">MTD:</span> $'+(pnl.month_pnl||0).toFixed(2)+'</div>'+
      '<div><span class="label">YTD:</span> $'+(pnl.year_pnl||0).toFixed(2)+'</div>'+
      '<div><span class="label">Open positions:</span> '+(pnl.open_positions||0)+'</div>'+
      '<div><span class="label">Win rate MTD:</span> '+(pnl.win_rate_mtd||0).toFixed(0)+'%</div>';
    // Farm
    let f=d.farm||{};
    document.getElementById('farm').innerHTML=
      '<div><span class="label">Total configs:</span> '+(f.total_configs||0).toLocaleString()+'</div>'+
      '<div><span class="label">Active traders:</span> '+(f.active_traders||0).toLocaleString()+'</div>'+
      '<div><span class="label">Last eval:</span> '+(f.last_evaluation_symbol||'N/A')+'</div>';
    // Logs
    let logs=d.recent_logs||'No logs available';
    document.getElementById('logs').textContent=logs;
  }catch(e){console.error(e);}
}
async function runCmd(){
  let inp=document.getElementById('cmd-input');
  let cmd=inp.value.trim();if(!cmd)return;
  document.getElementById('cmd-output').textContent='Running...';
  try{
    let r=await fetch('/api/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({command:cmd})});
    let d=await r.json();
    document.getElementById('cmd-output').textContent=d.result||d.error||'No output';
  }catch(e){document.getElementById('cmd-output').textContent='Error: '+e;}
}
load();setInterval(load,30000);
</script>
</body></html>"""


class ArgusWebDashboard:
    """Lightweight local web dashboard for Argus."""

    def __init__(self, host: str = '127.0.0.1', port: int = 8777):
        self.host = host
        self.port = port
        self._app = web.Application()
        self._runner = None

        # Callbacks set by orchestrator
        self._get_status: Optional[Callable] = None
        self._get_pnl: Optional[Callable] = None
        self._get_farm_status: Optional[Callable] = None
        self._get_providers: Optional[Callable] = None
        self._run_command: Optional[Callable] = None
        self._get_recent_logs: Optional[Callable] = None
        self._boot_phases: str = ''

        self._app.router.add_get('/', self._handle_index)
        self._app.router.add_get('/api/status', self._handle_status)
        self._app.router.add_post('/api/command', self._handle_command)

    def set_callbacks(
        self,
        get_status=None,
        get_pnl=None,
        get_farm_status=None,
        get_providers=None,
        run_command=None,
        get_recent_logs=None,
    ):
        if get_status: self._get_status = get_status
        if get_pnl: self._get_pnl = get_pnl
        if get_farm_status: self._get_farm_status = get_farm_status
        if get_providers: self._get_providers = get_providers
        if run_command: self._run_command = run_command
        if get_recent_logs: self._get_recent_logs = get_recent_logs

    def set_boot_phases(self, text: str):
        self._boot_phases = text

    async def start(self):
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()
        logger.info(f"Dashboard running at http://{self.host}:{self.port}")

    async def stop(self):
        if self._runner:
            await self._runner.cleanup()

    async def _handle_index(self, request):
        return web.Response(text=_HTML, content_type='text/html')

    async def _handle_status(self, request):
        start = time.perf_counter()
        status_code = 200

        def _ensure_json_safe(payload: Dict[str, Any]) -> Dict[str, Any]:
            try:
                json.dumps(payload)
                return payload
            except TypeError:
                return json.loads(json.dumps(payload, default=str))

        result: Dict[str, Any] = {}
        try:
            # System info
            from ..core.logger import _uptime
            system = {'uptime': _uptime(), 'boot_phases': self._boot_phases}
            if self._get_status:
                try:
                    s = await self._get_status()
                    system.update(s)
                except Exception as e:
                    system['error'] = str(e)
            result['system'] = system

            # Providers
            if self._get_providers:
                try:
                    result['providers'] = await self._get_providers()
                except Exception as e:
                    result['providers'] = {'error': str(e)}

            # PnL
            if self._get_pnl:
                try:
                    result['pnl'] = await self._get_pnl()
                except Exception as e:
                    result['pnl'] = {'error': str(e)}

            # Farm
            if self._get_farm_status:
                try:
                    result['farm'] = await self._get_farm_status()
                except Exception as e:
                    result['farm'] = {'error': str(e)}

            # Recent logs
            if self._get_recent_logs:
                try:
                    result['recent_logs'] = await self._get_recent_logs()
                except Exception as e:
                    result['recent_logs'] = str(e)
        except Exception as e:
            logger.error(f"/api/status error: {e}")
            result = {'error': str(e)}
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.info(f"/api/status {status_code} in {duration_ms:.1f}ms")

        return web.json_response(_ensure_json_safe(result), status=status_code)

    async def _handle_command(self, request):
        start = time.perf_counter()
        status_code = 200
        try:
            data = await request.json()
            cmd = data.get('command', '').strip()
            if not cmd:
                response = {'error': 'No command provided'}
            elif self._run_command:
                result = await self._run_command(cmd)
                response = {'result': result}
            else:
                response = {'error': 'Command handler not configured'}
        except Exception as e:
            logger.error(f"/api/command error: {e}")
            response = {'error': str(e)}
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.info(f"/api/command {status_code} in {duration_ms:.1f}ms")

        return web.json_response(response, status=status_code)
