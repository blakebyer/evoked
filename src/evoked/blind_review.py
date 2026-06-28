import os
from datetime import datetime
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from evoked.io import load_bulk
from evoked.preprocess import preprocess
from evoked.base import PreprocessParams

st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
        /* Remove top and bottom margins */
        .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
        }
        
        /* Remove default spacing between containers */
        .stVerticalBlock {
            gap: 0px !important;
        }
        .stAppDeployButton { visibility: hidden; display: none; }
        header { visibility: hidden; }
        #MainMenu { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)
#st.title("Blind Trace Review")

if "auth" not in st.session_state:
    st.session_state.auth = False

with st.container(width="content", horizontal=True):
    if not st.session_state.auth:
        pw = st.text_input("Access code", placeholder="Access code", type="password", label_visibility="collapsed", width=200)
        if st.button("Enter"):
            if pw == st.secrets["APP_PASSWORD"]:
                st.session_state.auth = True
                st.rerun()
            else:
                st.warning("Wrong access code.")
        st.stop()

if "alias" not in st.session_state:
    st.session_state.alias = ""

with st.container(width="content", horizontal=True):
    if st.session_state.alias == "":
        alias = st.text_input("Reviewer Alias", placeholder="Reviewer Alias", label_visibility="collapsed", width=200)

        if st.button("Go"):
            if alias.strip():
                st.session_state.alias = alias.strip()
                st.rerun()
            else:
                st.warning("Enter reviewer alias.")

        st.stop()

# Settings
os.makedirs("benchmark", exist_ok=True)
safe_alias = st.session_state.alias.replace(" ", "_")
SAVE_PATH = os.path.join("benchmark", f"blind_classifier_{safe_alias}.csv")

REPNUM = 3
INT_LIST = [25, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600]

PARAMS = PreprocessParams(
    baseline_window=(0, 0.1),
    artifact_window=(0, 2.25),
    artifact="template",
    smoothing="savgol",
    smoothing_params={
        "size": 7,
        "window_length": 11,
        "polyorder": 3,
        "cutoff": 2000.0,
        "order": 2,
    },
)

files_list = [ 
    # 16 CA1
    '2025_03_04_0002.abf',
    '2025_03_03_0000.abf',
    '2025_03_05_0004.abf',
    '2025_03_06_0002.abf',
    '2025_03_06_0010.abf', 
    '2025_03_02_0000.abf', 
    '2025_03_07_0002.abf', 
    '2025_03_06_0004.abf', 
    '2025_03_05_0000.abf', 
    '2025_03_04_0012.abf',       
    '2025_03_03_0007.abf',
    '2025_03_07_0008.abf',
    '2025_03_04_0000.abf',
    '2025_05_22_0000.abf', 
    '2025_03_05_0002.abf',
    '2025_05_22_0006.abf',

    # 16 DG
    '2025_05_22_0005.abf',
    '2025_03_06_0001.abf',
    '2025_03_05_0007.abf',
    '2025_03_04_0003.abf',
    '2025_03_04_0005.abf',
    '2025_05_22_0001.abf',
    '2025_03_07_0009.abf',
    '2025_03_07_0005.abf',
    '2025_03_07_0007.abf',
    '2025_03_02_0010.abf',
    '2025_03_04_0009.abf',
    '2025_03_06_0007.abf',
    '2025_03_03_0001.abf',
    '2025_03_05_0009.abf',
    '2025_03_03_0006.abf',
    '2025_03_02_0003.abf'
]

base_path = os.path.join(os.path.dirname(__file__), "data")

# get all valid ABF files
all_files = [os.path.join(base_path, f) for f in files_list if f.endswith('.abf')]

# Load / preprocess
if "all_proc" not in st.session_state:
    all_raw = load_bulk(
        all_files,
        intensities=INT_LIST,
        repnum=REPNUM,
    )

    st.session_state.all_proc = preprocess(all_raw, PARAMS)

all_proc = st.session_state.all_proc

# random blinded trace order
if "trace_order" not in st.session_state:
    trace_order = (
        all_proc[["id", "intensity"]]
        .drop_duplicates()
        .sample(frac=1) # random_state=42
        .reset_index(drop=True)
    )

    trace_order["trace_idx"] = trace_order.index
    st.session_state.trace_order = trace_order

traces = st.session_state.trace_order

# load saved responses
if "responses" not in st.session_state:
    if os.path.exists(SAVE_PATH):
        df = pd.read_csv(SAVE_PATH)

        df = df[
            df["fiber_volley"].notna()
            & df["fepsp"].notna()
            & df["population_spike"].notna()
            & (df["fiber_volley"] != "")
            & (df["fepsp"] != "")
            & (df["population_spike"] != "")
        ]

        st.session_state.responses = {
            int(row["trace_idx"]): {
                "fiber_volley": row["fiber_volley"],
                "fepsp": row["fepsp"],
                "population_spike": row["population_spike"],
            }
            for _, row in df.iterrows()
        }

        saved_idxs = set(st.session_state.responses.keys())

        st.session_state.idx = next(
            (i for i in range(len(traces)) if i not in saved_idxs),
            len(traces) - 1,
        )

    else:
        st.session_state.responses = {}
        st.session_state.idx = 0

# current trace
idx = st.session_state.idx

trace_info = traces.iloc[idx]
trace_id = trace_info["id"]
trace_intensity = trace_info["intensity"]

trace_df = all_proc[
    (all_proc["id"] == trace_id)
    & (all_proc["intensity"] == trace_intensity)
].sort_values("time")

alias = st.session_state.alias

st.caption(f"Trace {idx + 1} / {len(traces)}")

# plotly trace viewer
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=trace_df["time"] * 1000,
        y=trace_df["voltage"],
        mode="lines",
    )
)

fig.update_layout(
    height=425,
    margin=dict(l=0, r=0, t=0, b=0),
    xaxis_title="Time (ms)",
    yaxis_title="Voltage (mV)",
)

st.plotly_chart(fig, width="stretch")

st.caption(
    "Mark **Yes** only if you would report a reliable quantitative value for this feature. "
    "Mark **No** if the feature is absent, noisy, ambiguous, or would require guessing."
)

# load previous answer
current = st.session_state.responses.get(idx, {})

# questions
col1, col2, col3 = st.columns(3)

options = ["Unanswered", "Yes", "No"]

def answer_index(value):
    return options.index(value) if value in options else 0

current = st.session_state.responses.get(idx, {})

col1, col2, col3 = st.columns(3)

with col1:
    fv = st.radio(
        "Fiber volley measurable?",
        options,
        index=answer_index(current.get("fiber_volley", "Unanswered")),
        horizontal=True,
        key=f"fv_{idx}",
    )

with col2:
    fepsp = st.radio(
        "fEPSP slope measurable?",
        options,
        index=answer_index(current.get("fepsp", "Unanswered")),
        horizontal=True,
        key=f"fepsp_{idx}",
    )

with col3:
    ps = st.radio(
        "Population spike measurable?",
        options,
        index=answer_index(current.get("population_spike", "Unanswered")),
        horizontal=True,
        key=f"ps_{idx}",
    )


# save function
def save_response():
    if fv == "Unanswered" or fepsp == "Unanswered" or ps == "Unanswered":
        return False

    st.session_state.responses[idx] = {
        "fiber_volley": fv,
        "fepsp": fepsp,
        "population_spike": ps,
    }

    row = {
        "trace_idx": idx,
        "id": trace_id,
        "intensity": trace_intensity,
        "fiber_volley": fv,
        "fepsp": fepsp,
        "population_spike": ps,
        "reviewer_alias": alias,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    if os.path.exists(SAVE_PATH):
        df = pd.read_csv(SAVE_PATH)
        df = df[df["trace_idx"] != idx]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.sort_values("trace_idx").to_csv(SAVE_PATH, index=False)

    return True

# navigation
nav1, _, nav3 = st.columns(3)

with nav1:
    if st.button("Previous", disabled=idx == 0):
        save_response()
        st.session_state.idx -= 1
        st.rerun()

with nav3:
    if st.button("Save + Next", disabled=idx == len(traces) - 1):
        if save_response():
            st.session_state.idx += 1
            st.rerun()
        else:
            st.warning("Answer all questions before moving next.")

# optional debug
#st.write(st.session_state.responses)