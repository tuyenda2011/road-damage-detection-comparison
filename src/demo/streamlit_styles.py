"""Visual system for the Streamlit dashboard — Light Mode."""

APP_CSS = r"""
<style>
/* ── Google Fonts ────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── Design tokens ───────────────────────────────────────────────── */
:root {
    --canvas:        #ffffff;
    --canvas-soft:   #f8fafb;
    --canvas-hover:  #f0f7f6;
    --panel:         #ffffff;
    --panel-tint:    #f2fbf9;
    --line:          #e2eaed;
    --line-mid:      #c8d8df;
    --text:          #14303f;
    --text-2:        #3a5561;
    --muted:         #6b8a97;
    --accent:        #0f9f8f;
    --accent-light:  rgba(15,159,143,0.10);
    --accent-mid:    rgba(15,159,143,0.22);
    --amber:         #b07700;
    --amber-bg:      rgba(176,119,0,0.09);
    --danger:        #c93c3c;
    --danger-bg:     rgba(201,60,60,0.09);
    --violet:        #6d28d9;
    --violet-bg:     rgba(109,40,217,0.09);
    --shadow-xs:     0 1px 3px rgba(14,35,50,0.06);
    --shadow-sm:     0 4px 12px rgba(14,35,50,0.08);
    --shadow-md:     0 8px 28px rgba(14,35,50,0.10);
    --shadow-lg:     0 16px 48px rgba(14,35,50,0.12);
    --radius-sm:     8px;
    --radius-md:     12px;
    --radius-lg:     18px;
    --radius-xl:     24px;
}

/* ── Base ────────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
    color-scheme: light;
    -webkit-font-smoothing: antialiased;
}

[data-testid="stAppViewContainer"] {
    background: var(--canvas-soft);
}

[data-testid="stHeader"] {
    background: rgba(255,255,255,0.85);
    backdrop-filter: blur(20px);
    border-bottom: 1px solid var(--line);
}

[data-testid="stMainBlockContainer"] {
    max-width: 1440px;
    padding-top: 1.5rem;
    padding-bottom: 5rem;
}

h1, h2, h3, h4 {
    color: var(--text);
    letter-spacing: -0.03em;
    font-weight: 700;
}

/* ── Sidebar ─────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffffff 0%, #f5fbfa 100%) !important;
    border-right: 1px solid var(--line) !important;
    box-shadow: 2px 0 20px rgba(14,35,50,0.04);
}
[data-testid="stSidebar"] .block-container {
    padding: 1.2rem 1.1rem 2.5rem;
}
[data-testid="stSidebar"] hr {
    border-color: var(--line) !important;
    margin: 0.9rem 0;
}

/* ── Brand lockup ────────────────────────────────────────────────── */
.brand-lockup {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    padding: 0.1rem 0 1rem;
}
.brand-mark {
    width: 40px;
    height: 40px;
    display: grid;
    place-items: center;
    border-radius: 11px;
    color: #fff;
    font-size: 1.2rem;
    font-weight: 700;
    background: linear-gradient(140deg, #2dd4bf 0%, #0c8a7c 100%);
    box-shadow: 0 6px 18px rgba(15,159,143,0.28);
    flex-shrink: 0;
}
.brand-name {
    color: var(--text);
    font-weight: 800;
    font-size: 0.9rem;
    line-height: 1.2;
    letter-spacing: -0.02em;
}
.brand-meta {
    color: var(--muted);
    font-size: 0.62rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 0.15rem;
}

/* ── Sidebar labels ──────────────────────────────────────────────── */
.side-label {
    color: var(--muted);
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin: 0.35rem 0 0.6rem;
}

/* ── Model context card ──────────────────────────────────────────── */
.model-context {
    border: 1px solid var(--line);
    border-radius: var(--radius-md);
    padding: 0.75rem 0.9rem;
    margin: 0.15rem 0 0.9rem;
    background: var(--panel-tint);
}
.model-context-title {
    font-size: 0.82rem;
    font-weight: 700;
    color: var(--text);
}
.model-context-copy {
    color: var(--muted);
    font-size: 0.72rem;
    margin-top: 0.25rem;
    line-height: 1.5;
}

/* ── Runtime rows ────────────────────────────────────────────────── */
.runtime-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.6rem;
    padding: 0.45rem 0;
    border-bottom: 1px solid var(--line);
    font-size: 0.73rem;
}
.runtime-row:last-child { border-bottom: none; }
.runtime-label { color: var(--muted); }
.runtime-value  { color: var(--text); font-weight: 600; text-align: right; }

/* ── Status dot ──────────────────────────────────────────────────── */
.status-dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    margin-right: 0.4rem;
    background: var(--danger);
    box-shadow: 0 0 0 3px var(--danger-bg);
}
.status-dot.ready {
    background: var(--accent);
    box-shadow: 0 0 0 3px var(--accent-mid);
}

/* ── Hero banner ─────────────────────────────────────────────────── */
.app-hero {
    position: relative;
    overflow: hidden;
    border: 1px solid var(--line);
    border-radius: var(--radius-xl);
    padding: 1.8rem 2rem 1.6rem;
    margin-bottom: 1.25rem;
    background: linear-gradient(120deg, #edfaf7 0%, #ffffff 55%, #f5f3ff 100%);
    box-shadow: var(--shadow-md);
}
.app-hero::before {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: var(--radius-xl);
    background: radial-gradient(ellipse 60% 120% at 90% -10%, rgba(15,159,143,0.12) 0%, transparent 60%);
    pointer-events: none;
}
.app-hero::after {
    content: '';
    position: absolute;
    width: 280px;
    height: 280px;
    right: -70px;
    top: -130px;
    border-radius: 50%;
    border: 1px solid rgba(15,159,143,0.14);
    box-shadow:
        0 0 0 40px rgba(15,159,143,0.04),
        0 0 0 80px rgba(15,159,143,0.025);
    pointer-events: none;
}
.hero-grid {
    position: relative;
    z-index: 1;
    display: grid;
    grid-template-columns: minmax(0, 1.5fr) minmax(300px, 0.8fr);
    gap: 2rem;
    align-items: end;
}
.hero-eyebrow, .section-eyebrow {
    color: var(--accent);
    font-size: 0.64rem;
    font-weight: 800;
    letter-spacing: 0.18em;
    text-transform: uppercase;
}
.hero-title {
    color: var(--text);
    font-size: clamp(1.7rem, 2.8vw, 2.5rem);
    font-weight: 800;
    letter-spacing: -0.05em;
    line-height: 1.07;
    margin: 0.4rem 0 0.5rem;
}
.hero-copy {
    color: var(--text-2);
    font-size: 0.88rem;
    line-height: 1.65;
    max-width: 640px;
}
.hero-stats {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.6rem;
}
.hero-stat {
    min-width: 0;
    border-left: 2px solid var(--accent-mid);
    padding-left: 0.75rem;
}
.hero-stat-value {
    color: var(--text);
    font-size: 1.2rem;
    font-weight: 800;
    letter-spacing: -0.03em;
}
.hero-stat-label {
    color: var(--muted);
    font-size: 0.63rem;
    margin-top: 0.15rem;
}

/* ── Section headers ─────────────────────────────────────────────── */
.section-heading  { margin: 0.8rem 0 1.1rem; }
.section-title    { color: var(--text); font-size: 1.18rem; font-weight: 700; margin-top: 0.2rem; }
.section-copy     { color: var(--muted); font-size: 0.8rem; margin-top: 0.28rem; max-width: 800px; line-height: 1.55; }

/* ── Metric cards ────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: var(--panel) !important;
    border: 1px solid var(--line) !important;
    border-radius: var(--radius-lg) !important;
    padding: 1rem 1.1rem !important;
    box-shadow: var(--shadow-xs) !important;
    transition: box-shadow 0.2s, transform 0.2s;
}
[data-testid="stMetric"]:hover {
    box-shadow: var(--shadow-sm) !important;
    transform: translateY(-1px);
}
[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em;
}
[data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-size: 1.5rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em;
}

/* ── Checkpoint cards ────────────────────────────────────────────── */
.checkpoint-card {
    min-height: 115px;
    padding: 1rem 1.05rem;
    border: 1px solid var(--line);
    border-radius: var(--radius-lg);
    background: var(--panel);
    box-shadow: var(--shadow-xs);
    transition: box-shadow 0.2s, transform 0.2s;
}
.checkpoint-card:hover {
    box-shadow: var(--shadow-sm);
    transform: translateY(-1px);
}
.checkpoint-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.5rem;
}
.checkpoint-name  { color: var(--text); font-weight: 700; font-size: 0.86rem; }
.checkpoint-state { font-size: 0.67rem; font-weight: 700; white-space: nowrap; }
.checkpoint-state.ready   { color: var(--accent); }
.checkpoint-state.missing { color: var(--danger); }
.checkpoint-copy  { color: var(--muted); font-size: 0.71rem; margin-top: 0.5rem; line-height: 1.5; }
.checkpoint-line  { height: 3px; border-radius: 99px; margin-top: 0.8rem; }

/* ── Workbench empty state ───────────────────────────────────────── */
.workbench-empty {
    border: 1.5px dashed var(--line-mid);
    border-radius: var(--radius-xl);
    padding: 2.5rem 1.5rem 2rem;
    text-align: center;
    background: linear-gradient(180deg, var(--panel-tint) 0%, #ffffff 100%);
}
.workbench-icon {
    width: 52px;
    height: 52px;
    display: grid;
    place-items: center;
    margin: 0 auto 0.85rem;
    border-radius: 14px;
    background: var(--accent-light);
    color: var(--accent);
    font-size: 1.5rem;
}
.workbench-title {
    color: var(--text);
    font-size: 1rem;
    font-weight: 700;
}
.workbench-copy {
    color: var(--muted);
    font-size: 0.8rem;
    margin-top: 0.35rem;
    max-width: 380px;
    margin-left: auto;
    margin-right: auto;
    line-height: 1.6;
}
.step-row { display: flex; justify-content: center; flex-wrap: wrap; gap: 0.45rem; margin-top: 1.1rem; }
.step-chip {
    color: var(--muted);
    font-size: 0.68rem;
    font-weight: 500;
    border: 1px solid var(--line);
    border-radius: 99px;
    padding: 0.3rem 0.65rem;
    background: #fff;
}

/* ── File & result toolbars ──────────────────────────────────────── */
.file-summary {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.6rem;
    padding: 0.7rem 0.95rem;
    border: 1px solid var(--line);
    border-radius: var(--radius-md);
    background: var(--canvas-soft);
    margin: 0.4rem 0 0.85rem;
}
.file-name { color: var(--text); font-size: 0.8rem; font-weight: 600; }
.file-meta { color: var(--muted); font-size: 0.7rem; }

.result-toolbar {
    display: flex;
    align-items: center;
    gap: 0;
    flex-wrap: wrap;
    padding: 0.6rem 0.9rem;
    border: 1px solid var(--line);
    border-radius: var(--radius-md);
    background: var(--canvas-soft);
    margin-bottom: 0.9rem;
}
.result-chip {
    color: var(--text-2);
    font-size: 0.7rem;
    padding-right: 0.65rem;
    margin-right: 0.65rem;
    border-right: 1px solid var(--line);
    line-height: 1.8;
}
.result-chip:last-child { border-right: none; margin-right: 0; padding-right: 0; }
.result-chip strong { color: var(--text); font-weight: 700; }

/* ── Buttons ─────────────────────────────────────────────────────── */
.stButton > button {
    min-height: 2.5rem;
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
    font-size: 0.83rem !important;
    letter-spacing: 0.01em;
    transition: all 0.18s ease !important;
}
.stButton > button[kind="primary"] {
    color: #fff !important;
    background: linear-gradient(135deg, #13b8a5 0%, #0a8c7e 100%) !important;
    border: none !important;
    box-shadow: 0 4px 14px rgba(15,159,143,0.30) !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 22px rgba(15,159,143,0.40) !important;
}
.stButton > button[kind="primary"]:active { transform: translateY(0) !important; }

.stButton > button[kind="secondary"] {
    color: var(--text-2) !important;
    background: #fff !important;
    border: 1.5px solid var(--line-mid) !important;
}
.stButton > button[kind="secondary"]:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background: var(--accent-light) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:disabled {
    color: #9ab0b8 !important;
    background: #eef3f5 !important;
    border-color: var(--line) !important;
    box-shadow: none !important;
    transform: none !important;
}

/* ── File uploader ───────────────────────────────────────────────── */
[data-testid="stFileUploader"] section {
    min-height: 100px;
    border: 1.5px dashed var(--line-mid) !important;
    border-radius: var(--radius-md) !important;
    background: var(--canvas-soft) !important;
    transition: border-color 0.18s, background 0.18s !important;
}
[data-testid="stFileUploader"] section:hover {
    border-color: var(--accent) !important;
    background: var(--panel-tint) !important;
}

/* ── Images ──────────────────────────────────────────────────────── */
[data-testid="stImage"] img {
    border-radius: var(--radius-md) !important;
    border: 1px solid var(--line) !important;
    box-shadow: var(--shadow-sm) !important;
}

/* ── DataFrame ───────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--line) !important;
    border-radius: var(--radius-md) !important;
    overflow: hidden !important;
    box-shadow: var(--shadow-xs) !important;
}

/* ── Alerts ──────────────────────────────────────────────────────── */
[data-testid="stAlert"] { border-radius: var(--radius-md) !important; }

/* ── Tabs ────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.3rem;
    padding: 0.3rem 0.35rem;
    border: 1px solid var(--line);
    border-radius: var(--radius-md);
    background: var(--canvas-soft);
}
.stTabs [data-baseweb="tab"] {
    min-height: 2.35rem;
    padding: 0 1rem;
    border-radius: var(--radius-sm);
    color: var(--muted);
    font-size: 0.8rem;
    font-weight: 600;
    transition: color 0.15s, background 0.15s;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--text); background: var(--canvas-hover); }
.stTabs [aria-selected="true"] {
    color: var(--text) !important;
    background: #fff !important;
    box-shadow: var(--shadow-xs) !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none; }

/* ── Native inputs ───────────────────────────────────────────────── */
[data-baseweb="select"] > div,
[data-baseweb="input"] > div {
    background-color: #fff !important;
    border-color: var(--line) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text) !important;
    transition: border-color 0.18s;
}
[data-baseweb="select"] > div:focus-within,
[data-baseweb="input"] > div:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-light) !important;
}
[data-baseweb="select"] *, [data-baseweb="input"] input { color: var(--text) !important; }

/* ── Slider ──────────────────────────────────────────────────────── */
[data-testid="stSlider"] [role="slider"] {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
}
[data-testid="stSlider"] > div > div > div > div {
    background: var(--accent) !important;
}

/* ── Expander ────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid var(--line) !important;
    border-radius: var(--radius-md) !important;
    overflow: hidden;
}
[data-testid="stExpander"] > div:first-child {
    background: var(--canvas-soft) !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
}

/* ── Spinner ─────────────────────────────────────────────────────── */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* ── Code blocks ─────────────────────────────────────────────────── */
code { color: #0d7a6e !important; background: var(--accent-light) !important; border-radius: 4px !important; }
pre code { color: inherit !important; background: transparent !important; }

/* ── Divider ─────────────────────────────────────────────────────── */
hr { border-color: var(--line) !important; margin: 1rem 0; }

/* ── Scrollbar ───────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--canvas-soft); }
::-webkit-scrollbar-thumb { background: var(--line-mid); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }

/* ── Responsive ──────────────────────────────────────────────────── */
@media (max-width: 900px) {
    [data-testid="stMainBlockContainer"] { padding-left: 1rem; padding-right: 1rem; }
    .hero-grid { grid-template-columns: 1fr; gap: 1.2rem; }
    .hero-stats { max-width: 500px; }
    .app-hero { padding: 1.3rem 1.2rem; }
}
</style>
"""
