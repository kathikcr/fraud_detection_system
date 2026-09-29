"""Local dashboard application shell served by the existing FastAPI app."""

from __future__ import annotations

from fastapi.responses import HTMLResponse


def dashboard_shell() -> HTMLResponse:
    """Return the directly accessible, login-free dashboard shell."""
    return HTMLResponse(_DASHBOARD_HTML)


_DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="theme-color" content="#101821">
  <title>Fraudwatch — Analytics</title>
  <style>
    :root { color-scheme: dark; --bg:#101821; --panel:#18232e; --panel2:#1d2a36; --line:#2a3946; --muted:#91a0ad; --text:#e9f0f4; --mint:#67e0b1; --blue:#80b8ff; --amber:#f3bd66; }
    * { box-sizing:border-box; }
    body { margin:0; background:radial-gradient(ellipse at 72% -15%,#1e3841 0,transparent 42%),var(--bg); color:var(--text); font:14px/1.5 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; }
    a { color:inherit; text-decoration:none; }
    .app { min-height:100vh; display:grid; grid-template-columns:248px minmax(0,1fr); }
    aside { padding:25px 16px; border-right:1px solid var(--line); background:#111b24cc; }
    .brand { display:flex; align-items:center; gap:11px; padding:0 10px 29px; font-size:17px; font-weight:700; letter-spacing:-.4px; }
    .mark { width:32px;height:32px;border-radius:11px;background:linear-gradient(145deg,#6ce4b6,#4eaaa8);display:grid;place-items:center;color:#10201e;font-weight:900; }
    .eyebrow { color:var(--muted); text-transform:uppercase; letter-spacing:1.2px; font-size:10px; font-weight:750; padding:0 11px; margin:10px 0 9px; }
    nav { display:grid; gap:5px; }
    nav a { padding:10px 11px; display:flex; align-items:center; gap:11px; border-radius:9px; color:#aab7c0; transition:.16s ease; }
    nav a:hover,nav a.active { color:#effbf6;background:#20342f; }
    nav a.active { box-shadow:inset 2px 0 var(--mint); }
    nav .icon { width:18px;text-align:center;color:#8ba0ad;font-size:15px; }
    .side-note { margin:35px 5px 0; padding:14px; border:1px solid var(--line); border-radius:12px; color:var(--muted); font-size:12px; }
    .side-note strong { display:block;color:#dce7ec;margin-bottom:5px;font-size:12px; }
    main { min-width:0;padding:26px clamp(20px,4vw,56px) 56px;max-width:1600px;width:100%;margin:0 auto; }
    .topline { display:flex;align-items:center;justify-content:space-between;gap:18px;padding-bottom:26px;border-bottom:1px solid var(--line); }
    .crumb { color:var(--muted);font-size:12px; }
    .local { display:flex;align-items:center;gap:8px;color:#b3c0c9;font-size:12px; }
    .dot { width:7px;height:7px;background:var(--mint);border-radius:50%;box-shadow:0 0 12px #67e0b188; }
    .hero { padding:38px 0 27px;display:flex;justify-content:space-between;align-items:end;gap:20px; }
    .kicker { color:var(--mint);font-weight:700;font-size:11px;letter-spacing:1.4px;text-transform:uppercase; }
    h1 { margin:8px 0 8px;font-size:clamp(29px,4vw,43px);line-height:1.1;letter-spacing:-1.6px; }
    .sub { color:var(--muted);margin:0;max-width:600px;font-size:14px; }
    .selector { padding:10px 13px;color:#b7c5cd;background:var(--panel);border:1px solid var(--line);border-radius:9px;font-size:12px;white-space:nowrap; }
    .selector b { color:var(--text);font-weight:600; }
    .notice { border:1px solid #3b564e;background:linear-gradient(100deg,#1b332e,#1a2a30);border-radius:13px;padding:17px 19px;display:flex;align-items:flex-start;gap:13px;margin:4px 0 22px; }
    .notice-icon { width:28px;height:28px;display:grid;place-items:center;border-radius:9px;color:var(--mint);background:#28483e;flex:0 0 auto; }
    .notice strong { display:block;margin-bottom:3px;font-size:13px; }
    .notice p { color:#aabcb8;margin:0;font-size:12px; }
    .grid { display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px; }
    .card { background:linear-gradient(145deg,#1a2631,#17222c);border:1px solid var(--line);border-radius:13px;padding:19px;min-height:151px; }
    .card-head { display:flex;justify-content:space-between;align-items:center;color:#c5d0d7;font-weight:650;font-size:13px; }
    .tag { font-size:10px;letter-spacing:.4px;color:#a7b6be;border:1px solid #394956;border-radius:20px;padding:3px 8px;font-weight:550; }
    .card p { color:var(--muted);font-size:12px;line-height:1.55;margin:13px 0 0; }
    .placeholder { height:33px;margin-top:18px;border-radius:6px;background:repeating-linear-gradient(135deg,#22313d,#22313d 8px,#1e2b36 8px,#1e2b36 16px);opacity:.7; }
    .bottom { margin-top:22px;padding:16px 18px;border:1px solid var(--line);border-radius:12px;color:var(--muted);font-size:12px;display:flex;justify-content:space-between;gap:20px; }
    .bottom a { color:var(--blue); }
    section { scroll-margin-top:20px; }
    @media(max-width:850px){.app{grid-template-columns:76px minmax(0,1fr)}aside{padding:20px 10px}.brand{justify-content:center;padding:0 0 28px}.brand-name,.eyebrow,.nav-label,.side-note{display:none}nav a{justify-content:center;padding:12px 6px}.grid{grid-template-columns:1fr 1fr}}
    @media(max-width:560px){.app{display:block}aside{border-right:0;border-bottom:1px solid var(--line);padding:10px 14px}.brand{display:none}nav{display:flex;overflow-x:auto}nav a{flex:0 0 auto;padding:9px 11px}.nav-label{display:inline}.hero{align-items:flex-start;flex-direction:column;padding-top:27px}.grid{grid-template-columns:1fr}.card{min-height:125px}.topline{padding-bottom:16px}.bottom{flex-direction:column}}
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <div class="brand"><span class="mark">F</span><span class="brand-name">fraudwatch</span></div>
      <div class="eyebrow">Workspace</div>
      <nav aria-label="Dashboard navigation">
        <a class="active" href="#overview" aria-current="page"><span class="icon">◫</span><span class="nav-label">Overview</span></a>
        <a href="#performance"><span class="icon">⌁</span><span class="nav-label">Model performance</span></a>
        <a href="#investigations"><span class="icon">⌕</span><span class="nav-label">Investigations</span></a>
        <a href="#data-quality"><span class="icon">▦</span><span class="nav-label">Data quality</span></a>
      </nav>
      <div class="side-note"><strong>Local research workspace</strong>Built for demonstration and academic analysis. Dataset results will remain separate.</div>
    </aside>
    <main id="overview">
      <div class="topline"><div class="crumb">Workspace <span aria-hidden="true">/</span> Overview</div><div class="local"><span class="dot"></span> Local environment</div></div>
      <header class="hero">
        <div><div class="kicker">Fraud analytics</div><h1>See the signal.<br>Understand the risk.</h1><p class="sub">A focused workspace for transaction patterns, model performance, and explainable fraud investigation.</p></div>
        <div class="selector"><b>Dataset</b> &nbsp; Choose in the next dashboard phase&nbsp;⌄</div>
      </header>
      <div class="notice"><div class="notice-icon">✳</div><div><strong>Dashboard foundation is ready</strong><p>The local application shell and navigation are in place. Analytics pages are being added incrementally; no sample KPI values are shown as live results.</p></div></div>
      <div class="grid">
        <section class="card" id="performance"><div class="card-head">Model performance <span class="tag">NEXT PHASE</span></div><p>Precision, recall, PR-AUC, ROC-AUC, threshold behavior, and confusion matrices for validated model results.</p><div class="placeholder" aria-hidden="true"></div></section>
        <section class="card" id="investigations"><div class="card-head">Transaction investigations <span class="tag">PLANNED</span></div><p>Submit a transaction, inspect its risk score, and review local SHAP contributors where supported.</p><div class="placeholder" aria-hidden="true"></div></section>
        <section class="card" id="data-quality"><div class="card-head">Data quality <span class="tag">PLANNED</span></div><p>Explore each dataset independently, with clear quality checks and the synthetic dataset labelled accordingly.</p><div class="placeholder" aria-hidden="true"></div></section>
      </div>
      <footer class="bottom"><span>Local demo · No sign-in required · No customer personal information displayed</span><span>Model scores are decision support, not calibrated guarantees.</span></footer>
    </main>
  </div>
  <script>
    document.querySelectorAll('nav a').forEach(link => link.addEventListener('click', () => {
      document.querySelectorAll('nav a').forEach(item => { item.classList.remove('active'); item.removeAttribute('aria-current'); });
      link.classList.add('active'); link.setAttribute('aria-current', 'page');
    }));
  </script>
</body>
</html>"""
