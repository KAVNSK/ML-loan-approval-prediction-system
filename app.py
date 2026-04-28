from io import BytesIO
import pandas as pd
import streamlit as st
import config
import inference


@st.cache_resource
def load_runtime():
    return inference.load_artifacts()


@st.cache_data
def read_uploaded_csv(raw_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(BytesIO(raw_bytes))


@st.cache_data
def get_category_options(feature_cols: dict, encoder_categories: tuple) -> dict[str, list[str]]:
    options = {}
    categorical = feature_cols["categorical"]
    for feature, categories in zip(categorical, encoder_categories):
        options[feature] = [""] + [str(item) for item in categories]
    return options


def parse_optional_float(raw_value: str):
    value = raw_value.strip()
    if value == "":
        return None
    return float(value)


def build_single_row_input(preprocessor, feature_cols: dict) -> pd.DataFrame:
    numeric = feature_cols["numeric"]
    categorical = feature_cols["categorical"]
    encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    encoder_categories = tuple(tuple(items.tolist()) for items in encoder.categories_)
    category_options = get_category_options(feature_cols, encoder_categories)

    st.subheader("Одиночный прогноз")
    st.caption("Заполните поля ниже и нажмите кнопку расчета.")
    row_data = {}
    invalid_fields = []

    with st.form("single_prediction_form"):
        col_left, col_right = st.columns(2)

        for idx, feature in enumerate(numeric):
            container = col_left if idx % 2 == 0 else col_right
            raw_value = container.text_input(feature, value="")
            try:
                row_data[feature] = parse_optional_float(raw_value)
            except ValueError:
                row_data[feature] = None
                invalid_fields.append(feature)

        for idx, feature in enumerate(categorical):
            container = col_left if idx % 2 == 0 else col_right
            selected = container.selectbox(feature, category_options[feature], index=0)
            row_data[feature] = selected if selected else None

        submitted = st.form_submit_button("Рассчитать прогноз")
        if not submitted:
            return pd.DataFrame()
        if invalid_fields:
            fields = ", ".join(invalid_fields)
            st.error(f"Поля должны быть числами: {fields}")
            return pd.DataFrame()

    ordered = inference.feature_order(feature_cols)
    return pd.DataFrame([{feature: row_data.get(feature) for feature in ordered}])


def render_batch_predict(preprocessor, model, feature_cols: dict):
    st.subheader("Пакетный прогноз по CSV")
    uploaded_file = st.file_uploader("Загрузите CSV", type=["csv"], key="batch_uploader")
    if not uploaded_file:
        return

    input_df = read_uploaded_csv(uploaded_file.getvalue())
    st.write("Предпросмотр входных данных:")
    st.dataframe(input_df.head(10), use_container_width=True)

    if st.button("Сделать пакетный прогноз", type="primary"):
        try:
            pred_df = inference.predict_dataframe(preprocessor, model, feature_cols, input_df)
        except ValueError as exc:
            st.error(str(exc))
            return

        result_df = pd.concat([input_df, pred_df], axis=1)
        st.success("Готово. Ниже первые строки результата.")
        st.dataframe(result_df.head(20), use_container_width=True)

        buffer = BytesIO()
        result_df.to_csv(buffer, index=False)
        st.download_button(
            "Скачать predictions.csv",
            data=buffer.getvalue(),
            file_name="predictions.csv",
            mime="text/csv",
        )


def render_diagnostics():
    st.subheader("Диагностические графики")
    image_names = [
        "eda_histograms.png",
        "eda_boxplots.png",
        "eda_correlation_heatmap.png",
        "model_comparison_barchart.png",
        "confusion_matrices.png",
        "roc_curves_all.png",
        "feature_importance.png",
    ]
    for image_name in image_names:
        image_path = config.OUTPUT_DIR / image_name
        if image_path.exists():
            st.image(str(image_path), caption=image_name)
        else:
            st.info(f"График '{image_name}' не найден. Сначала запустите pipeline.py")


def main():
    st.set_page_config(page_title="Loan Approval GUI", page_icon=":bar_chart:", layout="wide")
    st.title("GUI: прогноз одобрения займа")
    st.caption("Интерфейс использует сохраненные артефакты из папки models/.")

    if not inference.artifacts_ready():
        missing = "\n".join([f"- {path.name}" for path in inference.missing_artifacts()])
        st.error("Модели не найдены. Запустите `python pipeline.py`, затем обновите страницу.")
        st.code(missing)
        return

    preprocessor, model, feature_cols = load_runtime()

    tab_single, tab_batch, tab_diag = st.tabs(
        ["Одиночный прогноз", "Пакетный прогноз", "Графики"]
    )

    with tab_single:
        row_df = build_single_row_input(preprocessor, feature_cols)
        if not row_df.empty:
            label, proba = inference.predict_one(preprocessor, model, feature_cols, row_df)
            verdict = "Одобрено" if label == "Y" else "Не одобрено"
            st.success(f"Решение модели: {label} ({verdict})")
            if proba is not None:
                st.metric("Вероятность одобрения", f"{proba:.2%}")

    with tab_batch:
        render_batch_predict(preprocessor, model, feature_cols)

    with tab_diag:
        render_diagnostics()


if __name__ == "__main__":
    main()
