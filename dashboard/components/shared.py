"""Shared CSS/sidebar injection for all IndiaLexABSA pages.

Uses Streamlit's native CSS variables (--text-color, --background-color,
--secondary-background-color) so the entire app responds automatically
when the user switches theme via the native ⋮ Settings menu.
"""
from __future__ import annotations
import streamlit as st


# ──────────────────────────────────────────────────────────────
# Default session state (no dark_mode — handled by Streamlit)
# ──────────────────────────────────────────────────────────────
PAGE_DEFAULTS = {
    "uploaded_docs":   [],
    "all_sentences":   [],
    "processing_done": False,
    "demo_mode":       True,
}


def init_state() -> None:
    """Initialise session state with defaults (idempotent)."""
    for k, v in PAGE_DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v


def palette() -> dict:
    """Return colour tokens as CSS variable references.

    These work inside inline `style` attributes:
        f'<div style="color:{p["tx"]}">Hello</div>'
    becomes:
        <div style="color:var(--text-color)">Hello</div>

    When the native theme changes, these update automatically.
    """
    return {
        "bg":   "var(--background-color)",
        "card": "var(--secondary-background-color)",
        "bd":   "rgba(128,128,128,0.22)",
        "tx":   "var(--text-color)",
        "mu":   "var(--text-color)",   # used with opacity in HTML
        "sur":  "var(--secondary-background-color)",
    }


# ──────────────────────────────────────────────────────────────
# CSS Injection
# ──────────────────────────────────────────────────────────────
def inject_global_css() -> None:
    """Inject global CSS that uses Streamlit CSS variables for full theme support."""
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

*, html, body, [class*="css"] {
    box-sizing: border-box;
}
/* Font: apply ONLY via body inheritance — never override span/div directly,
   because Streamlit uses plain <span> for Material Symbol icon text. */
body, p, h1, h2, h3, h4, h5, h6, label, a, input, textarea, select, button {
    font-family: 'Inter', sans-serif;
}

/* ═══ HIDE ALL NATIVE SIDEBAR TOGGLE CONTROLS ═══════════════ */
[data-testid="collapsedControl"],
button[aria-label="Close sidebar"],
button[aria-label="Collapse sidebar"],
[data-testid="stSidebarCollapseButton"] {
    display: none !important;
    visibility: hidden !important;
    width: 0 !important; height: 0 !important;
    overflow: hidden !important;
    position: absolute !important;
}

/* ═══ CUSTOM SIDEBAR TOGGLE BUTTON ═══════════════════════════ */
#ilx-sidebar-toggle {
    position: fixed;
    top: 12px;
    left: 12px;
    z-index: 999999;
    width: 34px;
    height: 34px;
    border-radius: 8px;
    border: 1px solid rgba(128,128,128,0.25);
    background: var(--secondary-background-color);
    color: var(--text-color);
    font-size: 14px;
    font-weight: 700;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.2s ease;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    font-family: 'Inter', sans-serif;
    letter-spacing: -1px;
}
#ilx-sidebar-toggle:hover {
    background: rgba(46,196,182,0.15);
    border-color: rgba(46,196,182,0.4);
    color: #2EC4B6;
    box-shadow: 0 4px 14px rgba(46,196,182,0.25);
}

/* ═══ SIDEBAR ═══════════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: var(--secondary-background-color) !important;
    border-right: 1px solid rgba(128,128,128,0.12) !important;
    transition: all 0.3s cubic-bezier(0.4,0,0.2,1) !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 0.8rem 0.9rem 1rem 0.9rem !important;
}

/* Sidebar text uses theme colors */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div {
    color: var(--text-color);
}

/* Sidebar nav links — capitalize page names, add PAGES header */
[data-testid="stSidebarNav"] {
    padding-top: 0 !important;
    margin-top: -0.5rem !important;
}
[data-testid="stSidebarNav"]::before {
    content: "PAGES";
    display: block;
    font-size: 0.57rem;
    font-weight: 700;
    color: var(--text-color);
    opacity: 0.35;
    letter-spacing: 1.3px;
    padding: 0 0.5rem 0.35rem;
}
[data-testid="stSidebarNav"] a {
    border-radius: 8px !important;
    transition: background 0.15s !important;
}
[data-testid="stSidebarNav"] a:hover {
    background: rgba(46,196,182,0.1) !important;
}
[data-testid="stSidebarNav"] a span {
    text-transform: capitalize !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.2px !important;
}

/* Sidebar st.button → ghost pill */
[data-testid="stSidebar"] .stButton > button {
    background: rgba(128,128,128,0.08) !important;
    border: 1px solid rgba(128,128,128,0.15) !important;
    border-radius: 8px !important;
    color: var(--text-color) !important;
    font-size: 0.88rem !important;
    padding: 0.38rem 1rem !important;
    width: 100% !important;
    min-height: 0 !important;
    box-shadow: none !important;
    transition: all 0.15s !important;
    text-align: left !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(46,196,182,0.12) !important;
    border-color: rgba(46,196,182,0.3) !important;
    color: #2EC4B6 !important;
    transform: none !important;
    box-shadow: none !important;
}

/* App background */
.stApp,
section[data-testid="stMain"],
section[data-testid="stMain"] > div {
    background: var(--background-color) !important;
    transition: background-color 0.3s !important;
}
.block-container {
    padding: 2rem 2.2rem 3rem 2.2rem !important;
    max-width: 100% !important;
    background: var(--background-color) !important;
}

/* Transparent header — blends into page background */
[data-testid="stHeader"] {
    background: transparent !important;
    background-color: transparent !important;
}

/* Hide ONLY the external Deploy button — keep everything else */
.stAppDeployButton,
[data-testid="stDeployButton"] {
    display: none !important;
}

/* Default teal button (main area) */
section[data-testid="stMain"] .stButton > button {
    background: linear-gradient(135deg, #2EC4B6, #1AADA0) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.65rem 1.5rem !important;
    box-shadow: 0 2px 10px rgba(46,196,182,0.3) !important;
    transition: all 0.2s !important;
}
section[data-testid="stMain"] .stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 5px 18px rgba(46,196,182,0.4) !important;
}

/* File uploader */
[data-testid="stFileUploaderDropzone"] {
    border: 2px dashed rgba(128,128,128,0.3) !important;
    border-radius: 14px !important;
    background: var(--secondary-background-color) !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #2EC4B6 !important;
}
/* Fix upload button label overlap */
[data-testid="stFileUploader"] label {
    font-size: 0 !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
}
[data-testid="stFileUploader"] section {
    margin-top: 0 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: var(--secondary-background-color) !important;
    border-radius: 10px !important;
    padding: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 7px !important;
    font-weight: 500 !important;
    color: var(--text-color) !important;
    opacity: 0.6;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    background: var(--background-color) !important;
    color: var(--text-color) !important;
    opacity: 1;
}

/* Misc widgets */
[data-testid="stExpander"] {
    border: 1px solid rgba(128,128,128,0.22) !important;
    border-radius: 12px !important;
    background: var(--secondary-background-color) !important;
}
[data-testid="stMetric"] {
    background: var(--secondary-background-color) !important;
    border: 1px solid rgba(128,128,128,0.22) !important;
    border-radius: 12px !important;
    padding: 1rem 1.2rem !important;
}
[data-testid="stAlert"] { border-radius: 12px !important; }
.stProgress > div > div > div { background: #2EC4B6 !important; }
.stSelectbox > div > div {
    background: var(--secondary-background-color) !important;
    border-color: rgba(128,128,128,0.22) !important;
    border-radius: 8px !important;
}

/* Sentence/clause cards */
.s-card {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,0.22);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.7rem;
    border-left: 4px solid transparent;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}
.s-text { font-size: 0.88rem; color: var(--text-color); line-height: 1.65; margin-bottom: 0.5rem; }
.s-meta { display: flex; gap: 0.6rem; align-items: center; flex-wrap: wrap; }
.s-badge {
    border-radius: 6px; padding: 2px 9px;
    font-size: 0.70rem; font-weight: 700;
}
.s-clause {
    background: rgba(59,130,246,0.12);
    color: #60A5FA;
    border-radius: 6px; padding: 2px 9px;
    font-size: 0.70rem; font-weight: 600;
}
.s-sub { font-size: 0.70rem; color: var(--text-color); opacity: 0.5; }

/* ═══ MUTED TEXT utility ═══ */
.ilx-muted { color: var(--text-color); opacity: 0.55; }
</style>

<!-- Custom sidebar toggle button -->
<div id="ilx-sidebar-toggle" title="Toggle sidebar">&laquo;</div>
""", unsafe_allow_html=True)

    # JavaScript for toggle — must use components.html() because
    # st.markdown strips <script> tags and shows the JS as raw text.
    import streamlit.components.v1 as components
    components.html("""
<script>
(function() {
    var doc = parent.document;
    function initToggle() {
        var btn = doc.getElementById('ilx-sidebar-toggle');
        var sidebar = doc.querySelector('[data-testid="stSidebar"]');
        if (!btn || !sidebar) { setTimeout(initToggle, 300); return; }

        var sidebarW = sidebar.getBoundingClientRect().width || 262;
        btn.style.left = sidebarW + 'px';

        if (btn._ilxBound) return;
        btn._ilxBound = true;

        btn.addEventListener('click', function() {
            if (sidebar.classList.contains('ilx-collapsed')) {
                sidebar.classList.remove('ilx-collapsed');
                sidebar.style.transform = 'none';
                sidebar.style.visibility = 'visible';
                sidebar.style.position = '';
                btn.innerHTML = '\u00AB';
                btn.style.left = sidebarW + 'px';
            } else {
                sidebar.classList.add('ilx-collapsed');
                sidebar.style.transform = 'translateX(-100%)';
                btn.innerHTML = '\u00BB';
                btn.style.left = '12px';
            }
        });
    }
    setTimeout(initToggle, 400);
})();
</script>
""", height=0)


# ──────────────────────────────────────────────────────────────
# Sidebar content
# ──────────────────────────────────────────────────────────────
def render_sidebar_content() -> None:
    """Render the sidebar: logo, demo toggle, session stats."""
    with st.sidebar:
        # Logo
        st.markdown("""
        <div style="padding:0.3rem 0 0.75rem;">
            <div style="font-size:1.35rem;font-weight:800;letter-spacing:-0.5px;
                        color:var(--text-color);">
                ⚖️ IndiaLex
            </div>
        </div>
        <hr style="border:none;border-top:1px solid rgba(128,128,128,0.15);margin:0 0 0.6rem;">
        """, unsafe_allow_html=True)

        # Demo mode toggle
        st.session_state.demo_mode = st.toggle(
            "Demo Mode",
            value=st.session_state.demo_mode,
            help="Use preloaded sample data.",
        )


# ──────────────────────────────────────────────────────────────
# Page setup  (single entry-point for every page)
# ──────────────────────────────────────────────────────────────
def page_setup(page_title: str, page_icon: str, current_page: str = "") -> dict:
    """
    Full page setup: init state, set_page_config, inject CSS, render sidebar.
    Returns palette dict for use in the page.
    """
    init_state()

    st.set_page_config(
        page_title=page_title,
        page_icon=page_icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_global_css()
    render_sidebar_content()
    return palette()
