import io
import json
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="AI-Assisted Data Wrangler & Visualizer", layout="wide")
sns.set_theme(style="whitegrid")


# -----------------------------
# Helpers
# -----------------------------
@st.cache_data
def load_data(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    if file_name.lower().endswith(".csv"):
        return pd.read_csv(io.BytesIO(file_bytes))
    if file_name.lower().endswith(".xlsx"):
        return pd.read_excel(io.BytesIO(file_bytes))
    if file_name.lower().endswith(".json"):
        return pd.read_json(io.BytesIO(file_bytes))
    raise ValueError("Unsupported file type. Please upload CSV, XLSX, or JSON.")


@st.cache_data
def profile_df(df: pd.DataFrame) -> dict:
    missing_count = df.isna().sum()
    missing_pct = (missing_count / len(df) * 100).round(2) if len(df) else missing_count
    missing_df = pd.DataFrame({
        "column": df.columns,
        "missing_count": missing_count.values,
        "missing_pct": missing_pct.values
    })
    return {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str),
        "missing": missing_df,
        "duplicates": int(df.duplicated().sum()),
        "numeric_summary": df.describe(include=[np.number]).T if not df.select_dtypes(include=[np.number]).empty else pd.DataFrame(),
        "categorical_summary": df.describe(include=["object", "category", "bool"]).T if not df.select_dtypes(include=["object", "category", "bool"]).empty else pd.DataFrame(),
    }


def init_state():
    defaults = {
        "original_df": None,
        "working_df": None,
        "log": [],
        "snapshots": [],
        "validation_violations": pd.DataFrame(),
        "uploaded_name": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_session():
    st.session_state.original_df = None
    st.session_state.working_df = None
    st.session_state.log = []
    st.session_state.snapshots = []
    st.session_state.validation_violations = pd.DataFrame()
    st.session_state.uploaded_name = None


def log_step(operation: str, params: dict, affected_columns=None):
    if affected_columns is None:
        affected_columns = []
    st.session_state.log.append({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "operation": operation,
        "params": params,
        "affected_columns": affected_columns,
    })


def push_snapshot():
    if st.session_state.working_df is not None:
        st.session_state.snapshots.append(st.session_state.working_df.copy())


def undo_last():
    if st.session_state.snapshots:
        st.session_state.working_df = st.session_state.snapshots.pop()
        if st.session_state.log:
            st.session_state.log.pop()


def safe_numeric_clean(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True),
        errors="coerce"
    )


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="cleaned_data")
    return output.getvalue()


def filtered_df_for_chart(df: pd.DataFrame):
    filtered = df.copy()

    st.subheader("Chart Filters")

    cat_cols = filtered.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = filtered.select_dtypes(include=[np.number]).columns.tolist()

    col1, col2 = st.columns(2)

    with col1:
        if cat_cols:
            filter_cat_col = st.selectbox("Category filter column", ["None"] + cat_cols, key="chart_cat_filter_col")
            if filter_cat_col != "None":
                vals = filtered[filter_cat_col].dropna().astype(str).unique().tolist()
                selected_vals = st.multiselect("Choose category values", vals, default=vals[:5], key="chart_cat_filter_vals")
                if selected_vals:
                    filtered = filtered[filtered[filter_cat_col].astype(str).isin(selected_vals)]

    with col2:
        if num_cols:
            filter_num_col = st.selectbox("Numeric range filter column", ["None"] + num_cols, key="chart_num_filter_col")
            if filter_num_col != "None" and not filtered[filter_num_col].dropna().empty:
                min_v = float(filtered[filter_num_col].min())
                max_v = float(filtered[filter_num_col].max())
                selected_range = st.slider("Numeric range", min_v, max_v, (min_v, max_v), key="chart_num_filter_range")
                filtered = filtered[(filtered[filter_num_col] >= selected_range[0]) & (filtered[filter_num_col] <= selected_range[1])]

    return filtered


# -----------------------------
# Main
# -----------------------------
init_state()

st.title("AI-Assisted Data Wrangler & Visualizer")

page = st.sidebar.radio(
    "Navigation",
    ["Page A — Upload & Overview", "Page B — Cleaning & Preparation Studio", "Page C — Visualization Builder", "Page D — Export & Report"]
)

st.sidebar.markdown("---")
if st.sidebar.button("Reset session"):
    reset_session()
    st.rerun()

if st.sidebar.button("Undo last step"):
    undo_last()
    st.rerun()

if st.session_state.working_df is not None:
    st.sidebar.success(f"Working dataset: {st.session_state.working_df.shape[0]} rows × {st.session_state.working_df.shape[1]} cols")


# -----------------------------
# Page A
# -----------------------------
if page == "Page A — Upload & Overview":
    st.header("Upload & Overview")

    uploaded_file = st.file_uploader("Upload CSV, XLSX, or JSON", type=["csv", "xlsx", "json"])

    if uploaded_file is not None:
        try:
            file_bytes = uploaded_file.getvalue()
            df = load_data(uploaded_file.name, file_bytes)
            st.session_state.original_df = df.copy()
            st.session_state.working_df = df.copy()
            st.session_state.uploaded_name = uploaded_file.name
            st.session_state.log = []
            st.session_state.snapshots = []
            st.success(f"Loaded {uploaded_file.name}")
        except Exception as e:
            st.error(f"Could not load file: {e}")

    if st.session_state.working_df is not None:
        df = st.session_state.working_df
        prof = profile_df(df)

        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", prof["shape"][0])
        c2.metric("Columns", prof["shape"][1])
        c3.metric("Duplicates", prof["duplicates"])

        st.subheader("Column names & inferred dtypes")
        dtype_df = pd.DataFrame({"column": df.columns, "dtype": df.dtypes.astype(str).values})
        st.dataframe(dtype_df, use_container_width=True)

        st.subheader("Missing values by column")
        st.dataframe(prof["missing"], use_container_width=True)

        st.subheader("Basic summary stats — numeric")
        if not prof["numeric_summary"].empty:
            st.dataframe(prof["numeric_summary"], use_container_width=True)
        else:
            st.info("No numeric columns found.")

        st.subheader("Basic summary stats — categorical")
        if not prof["categorical_summary"].empty:
            st.dataframe(prof["categorical_summary"], use_container_width=True)
        else:
            st.info("No categorical columns found.")

        st.subheader("Preview")
        st.dataframe(df.head(20), use_container_width=True)
    else:
        st.info("Upload a file first.")


# -----------------------------
# Page B
# -----------------------------
elif page == "Page B — Cleaning & Preparation Studio":
    st.header("Cleaning & Preparation Studio")

    if st.session_state.working_df is None:
        st.warning("Upload a dataset first on Page A.")
        st.stop()

    df = st.session_state.working_df

    st.subheader("Current data preview")
    st.dataframe(df.head(10), use_container_width=True)

    # 4.1 Missing Values
    with st.expander("4.1 Missing Values", expanded=False):
        missing_table = pd.DataFrame({
            "column": df.columns,
            "missing_count": df.isna().sum().values,
            "missing_pct": ((df.isna().sum() / len(df)) * 100).round(2).values if len(df) else 0
        })
        st.dataframe(missing_table, use_container_width=True)

        action = st.selectbox(
            "Missing value action",
            [
                "Drop rows with missing values in selected columns",
                "Drop columns above missing threshold %",
                "Fill selected columns with constant",
                "Fill selected numeric columns with mean",
                "Fill selected numeric columns with median",
                "Fill selected columns with mode/most frequent",
                "Forward fill selected columns",
                "Backward fill selected columns",
            ],
        )

        selected_cols = st.multiselect("Selected columns", df.columns.tolist(), default=df.columns.tolist()[:1])
        threshold = st.slider("Threshold %", 0, 100, 50)
        constant_value = st.text_input("Constant value")

        if st.button("Apply missing value action"):
            before_rows, before_cols = df.shape
            new_df = df.copy()
            push_snapshot()

            try:
                if action == "Drop rows with missing values in selected columns":
                    if selected_cols:
                        new_df = new_df.dropna(subset=selected_cols)
                elif action == "Drop columns above missing threshold %":
                    pct = new_df.isna().mean() * 100
                    cols_to_drop = pct[pct > threshold].index.tolist()
                    new_df = new_df.drop(columns=cols_to_drop)
                    selected_cols = cols_to_drop
                elif action == "Fill selected columns with constant":
                    new_df[selected_cols] = new_df[selected_cols].fillna(constant_value)
                elif action == "Fill selected numeric columns with mean":
                    for col in selected_cols:
                        if pd.api.types.is_numeric_dtype(new_df[col]):
                            new_df[col] = new_df[col].fillna(new_df[col].mean())
                elif action == "Fill selected numeric columns with median":
                    for col in selected_cols:
                        if pd.api.types.is_numeric_dtype(new_df[col]):
                            new_df[col] = new_df[col].fillna(new_df[col].median())
                elif action == "Fill selected columns with mode/most frequent":
                    for col in selected_cols:
                        if not new_df[col].mode().empty:
                            new_df[col] = new_df[col].fillna(new_df[col].mode().iloc[0])
                elif action == "Forward fill selected columns":
                    new_df[selected_cols] = new_df[selected_cols].ffill()
                elif action == "Backward fill selected columns":
                    new_df[selected_cols] = new_df[selected_cols].bfill()

                st.session_state.working_df = new_df
                log_step("missing_values", {"action": action, "threshold": threshold, "constant": constant_value}, selected_cols)
                st.success(f"Before: {before_rows}x{before_cols} → After: {new_df.shape[0]}x{new_df.shape[1]}")
            except Exception as e:
                st.error(f"Error applying missing value action: {e}")

    # 4.2 Duplicates
    with st.expander("4.2 Duplicates", expanded=False):
        dup_mode = st.radio("Detect duplicates by", ["Full row", "Subset of columns"])
        subset_cols = []
        if dup_mode == "Subset of columns":
            subset_cols = st.multiselect("Choose key columns", df.columns.tolist())

        try:
            dup_mask = df.duplicated(subset=subset_cols if subset_cols else None, keep=False)
            dup_df = df[dup_mask]
            st.write(f"Duplicate rows found: {len(dup_df)}")
            st.dataframe(dup_df.head(100), use_container_width=True)
        except Exception as e:
            st.error(f"Error detecting duplicates: {e}")

        keep_option = st.selectbox("If removing duplicates, keep", ["first", "last"])
        if st.button("Remove duplicates"):
            try:
                push_snapshot()
                new_df = df.drop_duplicates(subset=subset_cols if subset_cols else None, keep=keep_option)
                st.session_state.working_df = new_df
                log_step("remove_duplicates", {"keep": keep_option}, subset_cols)
                st.success(f"Removed duplicates. New shape: {new_df.shape}")
            except Exception as e:
                st.error(f"Error removing duplicates: {e}")

    # 4.3 Data Types & Parsing
    with st.expander("4.3 Data Types & Parsing", expanded=False):
        col = st.selectbox("Column to convert", df.columns.tolist(), key="dtype_col")
        target_type = st.selectbox("Target type", ["numeric", "categorical", "datetime"], key="dtype_target")
        datetime_format = st.text_input("Datetime format (optional)", value="", help="Example: %Y-%m-%d")

        if st.button("Convert column type"):
            try:
                push_snapshot()
                new_df = df.copy()

                if target_type == "numeric":
                    new_df[col] = safe_numeric_clean(new_df[col])
                elif target_type == "categorical":
                    new_df[col] = new_df[col].astype("category")
                elif target_type == "datetime":
                    if datetime_format.strip():
                        new_df[col] = pd.to_datetime(new_df[col], format=datetime_format, errors="coerce")
                    else:
                        new_df[col] = pd.to_datetime(new_df[col], errors="coerce")

                st.session_state.working_df = new_df
                log_step("convert_type", {"target_type": target_type, "datetime_format": datetime_format}, [col])
                st.success(f"Converted {col} to {target_type}")
            except Exception as e:
                st.error(f"Conversion failed: {e}")

    # 4.4 Categorical Data Tools
    with st.expander("4.4 Categorical Data Tools", expanded=False):
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if not cat_cols:
            st.info("No categorical columns available.")
        else:
            cat_selected = st.multiselect("Categorical columns", cat_cols, default=cat_cols[:1])
            casing_action = st.selectbox("Standardization action", ["trim whitespace", "lower case", "title case"])
            mapping_text = st.text_area("Mapping dictionary as JSON", value='{"old_value":"new_value"}')
            rare_threshold = st.number_input("Rare category threshold (count below becomes Other)", min_value=1, value=5)
            encode_cols = st.multiselect("One-hot encode columns (optional)", cat_cols)

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                if st.button("Apply standardization"):
                    try:
                        push_snapshot()
                        new_df = df.copy()
                        for c in cat_selected:
                            new_df[c] = new_df[c].astype(str).str.strip()
                            if casing_action == "lower case":
                                new_df[c] = new_df[c].str.lower()
                            elif casing_action == "title case":
                                new_df[c] = new_df[c].str.title()

                        st.session_state.working_df = new_df
                        log_step("categorical_standardization", {"action": casing_action}, cat_selected)
                        st.success("Standardization applied.")
                    except Exception as e:
                        st.error(f"Error: {e}")

            with c2:
                if st.button("Apply mapping"):
                    try:
                        mapping_dict = json.loads(mapping_text)
                        push_snapshot()
                        new_df = df.copy()
                        for c in cat_selected:
                            new_df[c] = new_df[c].replace(mapping_dict)
                        st.session_state.working_df = new_df
                        log_step("categorical_mapping", {"mapping": mapping_dict}, cat_selected)
                        st.success("Mapping applied.")
                    except Exception as e:
                        st.error(f"Invalid mapping JSON or mapping failed: {e}")

            with c3:
                if st.button("Group rare categories"):
                    try:
                        push_snapshot()
                        new_df = df.copy()
                        for c in cat_selected:
                            counts = new_df[c].value_counts(dropna=False)
                            rare_vals = counts[counts < rare_threshold].index
                            new_df[c] = new_df[c].replace(rare_vals, "Other")
                        st.session_state.working_df = new_df
                        log_step("rare_category_grouping", {"threshold": rare_threshold}, cat_selected)
                        st.success("Rare categories grouped.")
                    except Exception as e:
                        st.error(f"Error: {e}")

            with c4:
                if st.button("Apply one-hot encoding"):
                    try:
                        push_snapshot()
                        new_df = pd.get_dummies(df.copy(), columns=encode_cols, drop_first=False)
                        st.session_state.working_df = new_df
                        log_step("one_hot_encoding", {}, encode_cols)
                        st.success("One-hot encoding applied.")
                    except Exception as e:
                        st.error(f"Error: {e}")

    # 4.5 Numeric Cleaning
    with st.expander("4.5 Numeric Cleaning", expanded=False):
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            st.info("No numeric columns available.")
        else:
            outlier_cols = st.multiselect("Numeric columns for outlier handling", num_cols, default=num_cols[:1])
            outlier_action = st.selectbox("Outlier action", ["cap/winsorize", "remove outlier rows", "do nothing"])
            lower_q = st.slider("Lower quantile", 0.0, 0.25, 0.01, 0.01)
            upper_q = st.slider("Upper quantile", 0.75, 1.0, 0.99, 0.01)

            summary_rows = []
            for c in outlier_cols:
                q1 = df[c].quantile(0.25)
                q3 = df[c].quantile(0.75)
                iqr = q3 - q1
                low = q1 - 1.5 * iqr
                high = q3 + 1.5 * iqr
                count = int(((df[c] < low) | (df[c] > high)).sum())
                summary_rows.append({"column": c, "outliers_iqr_count": count})
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

            if st.button("Apply numeric cleaning"):
                try:
                    push_snapshot()
                    new_df = df.copy()
                    rows_before = len(new_df)

                    if outlier_action == "cap/winsorize":
                        for c in outlier_cols:
                            lo = new_df[c].quantile(lower_q)
                            hi = new_df[c].quantile(upper_q)
                            new_df[c] = new_df[c].clip(lo, hi)

                    elif outlier_action == "remove outlier rows":
                        keep_mask = pd.Series(True, index=new_df.index)
                        for c in outlier_cols:
                            q1 = new_df[c].quantile(0.25)
                            q3 = new_df[c].quantile(0.75)
                            iqr = q3 - q1
                            low = q1 - 1.5 * iqr
                            high = q3 + 1.5 * iqr
                            keep_mask &= new_df[c].between(low, high) | new_df[c].isna()
                        new_df = new_df[keep_mask]

                    st.session_state.working_df = new_df
                    log_step("numeric_cleaning", {"action": outlier_action, "lower_q": lower_q, "upper_q": upper_q}, outlier_cols)
                    st.success(f"Applied. Rows before: {rows_before}, rows after: {len(new_df)}")
                except Exception as e:
                    st.error(f"Error: {e}")

    # 4.6 Scaling
    with st.expander("4.6 Normalization / Scaling", expanded=False):
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            st.info("No numeric columns available.")
        else:
            scale_cols = st.multiselect("Columns to scale", num_cols, default=num_cols[:1], key="scale_cols")
            scale_method = st.selectbox("Scaling method", ["min-max", "z-score"], key="scale_method")

            if scale_cols:
                before_stats = df[scale_cols].agg(["min", "max", "mean", "std"]).T
                st.write("Before")
                st.dataframe(before_stats, use_container_width=True)

            if st.button("Apply scaling"):
                try:
                    push_snapshot()
                    new_df = df.copy()
                    for c in scale_cols:
                        if scale_method == "min-max":
                            min_v = new_df[c].min()
                            max_v = new_df[c].max()
                            if pd.notna(min_v) and pd.notna(max_v) and max_v != min_v:
                                new_df[c] = (new_df[c] - min_v) / (max_v - min_v)
                        else:
                            mean_v = new_df[c].mean()
                            std_v = new_df[c].std()
                            if pd.notna(std_v) and std_v != 0:
                                new_df[c] = (new_df[c] - mean_v) / std_v

                    st.session_state.working_df = new_df
                    log_step("scaling", {"method": scale_method}, scale_cols)
                    st.success("Scaling applied.")
                    after_stats = new_df[scale_cols].agg(["min", "max", "mean", "std"]).T
                    st.write("After")
                    st.dataframe(after_stats, use_container_width=True)
                except Exception as e:
                    st.error(f"Error: {e}")

    # 4.7 Column Operations
    with st.expander("4.7 Column Operations", expanded=False):
        rename_old = st.selectbox("Rename column", ["None"] + df.columns.tolist())
        rename_new = st.text_input("New name")
        drop_cols = st.multiselect("Drop columns", df.columns.tolist())

        st.markdown("### Create new column")
        create_mode = st.selectbox("Mode", ["ratio colA/colB", "difference colA-colB", "log(colA)", "colA-mean(colA)", "binning"])
        new_col_name = st.text_input("New column name", value="new_column")
        colA = st.selectbox("Column A", df.columns.tolist(), key="cola")
        colB = st.selectbox("Column B", df.columns.tolist(), key="colb")
        bin_method = st.selectbox("Binning method", ["equal-width", "quantile"])
        bins_n = st.number_input("Number of bins", min_value=2, max_value=20, value=4)

        if st.button("Apply column operations"):
            try:
                push_snapshot()
                new_df = df.copy()

                if rename_old != "None" and rename_new.strip():
                    new_df = new_df.rename(columns={rename_old: rename_new.strip()})
                    log_step("rename_column", {"new_name": rename_new.strip()}, [rename_old])

                if drop_cols:
                    new_df = new_df.drop(columns=drop_cols)
                    log_step("drop_columns", {}, drop_cols)

                if new_col_name.strip():
                    if create_mode == "ratio colA/colB":
                        new_df[new_col_name] = pd.to_numeric(new_df[colA], errors="coerce") / pd.to_numeric(new_df[colB], errors="coerce")
                    elif create_mode == "difference colA-colB":
                        new_df[new_col_name] = pd.to_numeric(new_df[colA], errors="coerce") - pd.to_numeric(new_df[colB], errors="coerce")
                    elif create_mode == "log(colA)":
                        new_df[new_col_name] = np.log(pd.to_numeric(new_df[colA], errors="coerce").replace(0, np.nan))
                    elif create_mode == "colA-mean(colA)":
                        s = pd.to_numeric(new_df[colA], errors="coerce")
                        new_df[new_col_name] = s - s.mean()
                    elif create_mode == "binning":
                        s = pd.to_numeric(new_df[colA], errors="coerce")
                        if bin_method == "equal-width":
                            new_df[new_col_name] = pd.cut(s, bins=bins_n)
                        else:
                            new_df[new_col_name] = pd.qcut(s, q=bins_n, duplicates="drop")
                    log_step("create_column", {"mode": create_mode, "binning": bin_method, "bins": bins_n}, [colA, colB])

                st.session_state.working_df = new_df
                st.success("Column operations applied.")
            except Exception as e:
                st.error(f"Error: {e}")

    # 4.8 Validation Rules
    with st.expander("4.8 Data Validation Rules", expanded=False):
        rule_type = st.selectbox("Validation rule type", ["numeric range", "allowed categories", "non-null constraint"])
        violations = pd.DataFrame()

        if rule_type == "numeric range":
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                v_col = st.selectbox("Numeric column", num_cols, key="val_num_col")
                min_allowed = st.number_input("Min allowed", value=float(df[v_col].min()) if not df[v_col].dropna().empty else 0.0)
                max_allowed = st.number_input("Max allowed", value=float(df[v_col].max()) if not df[v_col].dropna().empty else 1.0)
                if st.button("Run numeric range check"):
                    mask = (~df[v_col].between(min_allowed, max_allowed)) & df[v_col].notna()
                    violations = df[mask].copy()
            else:
                st.info("No numeric columns.")

        elif rule_type == "allowed categories":
            cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
            if cat_cols:
                v_col = st.selectbox("Categorical column", cat_cols, key="val_cat_col")
                allowed = st.text_input("Allowed categories, comma-separated")
                if st.button("Run allowed categories check"):
                    allowed_list = [x.strip() for x in allowed.split(",") if x.strip()]
                    mask = ~df[v_col].astype(str).isin(allowed_list) & df[v_col].notna()
                    violations = df[mask].copy()
            else:
                st.info("No categorical columns.")

        else:
            nn_cols = st.multiselect("Columns that must not be null", df.columns.tolist(), key="nonnull_cols")
            if st.button("Run non-null check"):
                if nn_cols:
                    mask = df[nn_cols].isna().any(axis=1)
                    violations = df[mask].copy()

        if not violations.empty:
            st.session_state.validation_violations = violations
            st.error(f"Violations found: {len(violations)}")
            st.dataframe(violations.head(100), use_container_width=True)
        elif "validation_violations" in st.session_state and st.button("Clear violations table"):
            st.session_state.validation_violations = pd.DataFrame()

    st.subheader("Transformation Log")
    if st.session_state.log:
        st.dataframe(pd.DataFrame(st.session_state.log), use_container_width=True)
    else:
        st.info("No transformations logged yet.")


# -----------------------------
# Page C
# -----------------------------
elif page == "Page C — Visualization Builder":
    st.header("Visualization Builder")

    if st.session_state.working_df is None:
        st.warning("Upload a dataset first on Page A.")
        st.stop()

    df = st.session_state.working_df.copy()
    df = filtered_df_for_chart(df)

    if df.empty:
        st.warning("No data left after filters.")
        st.stop()

    plot_type = st.selectbox(
        "Plot type",
        ["histogram", "box plot", "scatter plot", "line chart", "bar chart", "heatmap / correlation matrix"]
    )

    all_cols = df.columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    x_col = st.selectbox("X column", ["None"] + all_cols)
    y_col = st.selectbox("Y column", ["None"] + all_cols)
    color_col = st.selectbox("Optional color/group column", ["None"] + all_cols)
    agg = st.selectbox("Optional aggregation", ["None", "sum", "mean", "count", "median"])
    top_n = st.number_input("Top N categories for bar charts", min_value=3, max_value=50, value=10)

    fig, ax = plt.subplots(figsize=(10, 5))

    try:
        if plot_type == "histogram":
            if x_col != "None" and x_col in num_cols:
                ax.hist(df[x_col].dropna(), bins=30)
                ax.set_title(f"Histogram of {x_col}")
            else:
                st.warning("Choose a numeric X column.")

        elif plot_type == "box plot":
            if x_col != "None" and x_col in num_cols:
                ax.boxplot(df[x_col].dropna())
                ax.set_title(f"Box Plot of {x_col}")
                ax.set_xticklabels([x_col])
            else:
                st.warning("Choose a numeric X column.")

        elif plot_type == "scatter plot":
            if x_col in num_cols and y_col in num_cols:
                if color_col != "None" and color_col in cat_cols:
                    sns.scatterplot(data=df, x=x_col, y=y_col, hue=color_col, ax=ax)
                else:
                    sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
                ax.set_title(f"{y_col} vs {x_col}")
            else:
                st.warning("Choose numeric X and Y columns.")

        elif plot_type == "line chart":
            if x_col != "None" and y_col in num_cols:
                temp = df.copy()
                if np.issubdtype(temp[x_col].dtype, np.datetime64):
                    temp = temp.sort_values(x_col)
                if color_col != "None" and color_col in cat_cols:
                    sns.lineplot(data=temp, x=x_col, y=y_col, hue=color_col, ax=ax)
                else:
                    sns.lineplot(data=temp, x=x_col, y=y_col, ax=ax)
                ax.set_title(f"{y_col} over {x_col}")
            else:
                st.warning("Choose valid X and numeric Y.")

        elif plot_type == "bar chart":
            if x_col != "None":
                temp = df.copy()
                if agg != "None" and y_col != "None":
                    grouped = temp.groupby(x_col)[y_col].agg(agg).sort_values(ascending=False).head(top_n)
                else:
                    grouped = temp[x_col].value_counts().head(top_n)
                grouped.plot(kind="bar", ax=ax)
                ax.set_title(f"Bar Chart of {x_col}")
                ax.tick_params(axis="x", rotation=45)
            else:
                st.warning("Choose an X column.")

        elif plot_type == "heatmap / correlation matrix":
            if len(num_cols) >= 2:
                corr = df[num_cols].corr(numeric_only=True)
                sns.heatmap(corr, annot=True, cmap="Blues", ax=ax)
                ax.set_title("Correlation Matrix")
            else:
                st.warning("Need at least 2 numeric columns.")

        st.pyplot(fig)

    except Exception as e:
        st.error(f"Chart error: {e}")


# -----------------------------
# Page D
# -----------------------------
elif page == "Page D — Export & Report":
    st.header("Export & Report")

    if st.session_state.working_df is None:
        st.warning("Upload a dataset first on Page A.")
        st.stop()

    df = st.session_state.working_df
    report = {
        "uploaded_file": st.session_state.uploaded_name,
        "export_timestamp": datetime.now().isoformat(timespec="seconds"),
        "final_shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
        "steps": st.session_state.log,
    }

    st.subheader("Transformation Report")
    st.json(report)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    excel_bytes = to_excel_bytes(df)
    report_bytes = json.dumps(report, indent=2).encode("utf-8")
    recipe_bytes = json.dumps(st.session_state.log, indent=2).encode("utf-8")

    st.download_button(
        "Download cleaned dataset (CSV)",
        data=csv_bytes,
        file_name="cleaned_dataset.csv",
        mime="text/csv",
    )

    st.download_button(
        "Download cleaned dataset (Excel)",
        data=excel_bytes,
        file_name="cleaned_dataset.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.download_button(
        "Download transformation report (JSON)",
        data=report_bytes,
        file_name="transformation_report.json",
        mime="application/json",
    )

    st.download_button(
        "Download recipe / workflow log (JSON)",
        data=recipe_bytes,
        file_name="recipe.json",
        mime="application/json",
    )

    if not st.session_state.validation_violations.empty:
        st.subheader("Validation Violations")
        st.dataframe(st.session_state.validation_violations.head(100), use_container_width=True)
        st.download_button(
            "Download validation violations CSV",
            data=st.session_state.validation_violations.to_csv(index=False).encode("utf-8"),
            file_name="validation_violations.csv",
            mime="text/csv",
        )