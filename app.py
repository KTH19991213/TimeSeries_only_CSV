import subprocess
import sys

import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

from src.evaluate import compare_models
from src.io_utils import candidate_datetime_columns, candidate_numeric_columns, load_csv
from src.models import fit_and_predict
from src.preprocess import make_series, split_train_test
from src.visualize import plot_metric_bars, plot_residual_hist, plot_train_test_pred


def _is_empty_dataframe(obj) -> bool:
    return not isinstance(obj, pd.DataFrame) or obj.shape[0] == 0


def main() -> None:
    st.set_page_config(page_title="Time Series Forecast App", layout="wide")
    st.title("Time Series Forecast App")
    st.caption("과제용: CSV 업로드로 단변량 시계열 예측 + 평가지표 대시보드")

    df: pd.DataFrame | None = None

    uploaded = st.file_uploader("CSV 파일 업로드", type=["csv"])
    if uploaded is not None:
        try:
            df = load_csv(uploaded)
        except Exception as e:
            st.error(f"CSV 로딩 실패: {e}")
    else:
        st.info("CSV 파일을 선택하면 분석이 시작됩니다.")

    if _is_empty_dataframe(df):
        if df is not None:
            st.error("데이터프레임이 비어 있습니다.")
        st.stop()
        raise SystemExit

    with st.expander("원본 데이터 미리보기", expanded=False):
        if not isinstance(df, pd.DataFrame):
            st.error("유효한 데이터프레임이 없습니다.")
            st.stop()
            raise SystemExit
        st.dataframe(df.head(30), use_container_width=True)

    dt_cols = candidate_datetime_columns(df)
    num_cols = candidate_numeric_columns(df)

    if not dt_cols:
        st.error("날짜열을 인식하지 못했습니다. 날짜 문자열 컬럼을 확인하세요.")
        st.stop()
    if not num_cols:
        st.error("숫자열을 인식하지 못했습니다. 단변량 값 컬럼을 확인하세요.")
        st.stop()

    c_set1, c_set2, c_set3, c_set4 = st.columns(4)
    time_col = c_set1.selectbox("시간 열", dt_cols, index=0)
    value_col = c_set2.selectbox("값 열", num_cols, index=0)
    freq = c_set3.selectbox("빈도", ["auto", "D", "W", "M"], index=0)
    horizon = c_set4.number_input("시평(horizon)", min_value=3, max_value=180, value=24, step=1)

    max_test = max(4, min(180, len(df) - 1))
    test_size = st.slider("평가 구간(test_size)", min_value=3, max_value=max_test, value=min(36, max_test))

    try:
        y = make_series(df, time_col=time_col, value_col=value_col, freq=freq)
    except Exception as e:
        st.error(f"전처리 실패: {e}")
        st.stop()

    if len(y) < 20:
        st.warning("데이터 길이가 짧아 성능이 불안정할 수 있습니다. (권장 20 이상)")

    if test_size >= len(y):
        test_size = max(3, len(y) // 3)

    y_train, y_test = split_train_test(y, test_size=test_size)
    h_eff = min(int(horizon), len(y_test))
    if h_eff < int(horizon):
        st.info(f"현재 평가 구간 길이({len(y_test)})를 넘어 시평이 설정되어 시평을 {h_eff}로 자동 조정했습니다.")

    y_test_eval = y_test.iloc[:h_eff]
    pred_dict = fit_and_predict(y_train=y_train, y_test=y_test_eval, horizon=h_eff)

    if not pred_dict:
        st.error("예측 가능한 모델이 없습니다. 데이터 길이/빈도를 확인하세요.")
        st.stop()

    metric_df = compare_models(y_train=y_train, y_test=y_test_eval, pred_dict=pred_dict)
    if not isinstance(metric_df, pd.DataFrame) or metric_df.shape[0] == 0:
        st.error("평가지표 계산에 실패했습니다.")
        st.stop()

    best_model = metric_df.index[0]

    st.subheader("핵심 지표")
    m1, m2, m3, m4, m5 = st.columns(5)
    for i, key in enumerate(["MAE", "RMSE", "MAPE", "SMAPE", "MASE"]):
        [m1, m2, m3, m4, m5][i].metric(key, f"{metric_df.loc[best_model, key]:.4f}")

    st.subheader(f"Best Model: {best_model}")
    st.plotly_chart(plot_train_test_pred(y_train, y_test_eval, pred_dict), use_container_width=True)
    st.plotly_chart(plot_metric_bars(metric_df), use_container_width=True)

    st.subheader("모델별 지표 테이블")
    st.dataframe(metric_df, use_container_width=True)

    selected_model = st.selectbox("잔차 확인 모델", list(pred_dict.keys()), index=0)
    st.plotly_chart(plot_residual_hist(y_test_eval, pred_dict[selected_model]), use_container_width=True)

    st.success("파일/파라미터/시평 변경 시 자동으로 재예측됩니다.")


if __name__ == "__main__":
    if get_script_run_ctx() is None:
        subprocess.run([sys.executable, "-m", "streamlit", "run", __file__], check=False)
        raise SystemExit(0)
    main()
