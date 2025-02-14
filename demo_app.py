import streamlit as st
import os
import pandas as pd
import numpy as np
import pickle
import altair as alt
import seaborn as sns
import cv2
from PIL import Image

import matplotlib.pyplot as plt

from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    ConvLSTM2D,
    LSTM,
    Dense,
    Dropout,
    BatchNormalization,
    Flatten,
    TimeDistributed,
    Attention,
    Concatenate,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
from sklearn.metrics import confusion_matrix, classification_report

st.set_page_config(
    page_title="Hybrid Model Results",
    layout="wide",
)


########################
# Data Generator Class #
########################
class HybridDataGenerator(Sequence):
    def __init__(
        self, X_meteo, X_images, y, batch_size, class_weights=None, shuffle=True
    ):
        self.X_meteo = X_meteo.astype(np.float32)
        self.X_images = X_images.astype(np.float32)
        self.y = y.astype(np.float32)
        self.batch_size = batch_size
        self.class_weights = class_weights
        self.shuffle = shuffle
        self.indices = np.arange(len(self.X_meteo))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.X_meteo) / self.batch_size))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min(start + self.batch_size, len(self.X_meteo))
        batch_indices = self.indices[start:end]

        batch_X_meteo = self.X_meteo[batch_indices]
        batch_X_images = self.X_images[batch_indices]
        batch_y = self.y[batch_indices]

        if self.class_weights:
            batch_sample_weights = np.array(
                [self.class_weights[np.argmax(label)] for label in batch_y]
            )
        else:
            batch_sample_weights = np.ones(len(batch_y))

        return (
            {"meteo_input": batch_X_meteo, "cloud_input": batch_X_images},
            batch_y,
            batch_sample_weights,
        )

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


########################
# Model Definitions    #
########################
def hybrid_model_optimized(window_size, num_classes=4):
    image_steps = {24: 8, 48: 16, 72: 24}.get(window_size, 8)

    meteo_input = Input(shape=(window_size, 10), name="meteo_input")
    cloud_input = Input(shape=(image_steps, 64, 64, 1), name="cloud_input")

    def feature_extractor_images(input_layer):
        x = ConvLSTM2D(
            filters=8, kernel_size=(3, 3), padding="same", return_sequences=True
        )(input_layer)
        x = Dropout(0.25)(x)

        x = ConvLSTM2D(
            filters=16, kernel_size=(3, 3), padding="same", return_sequences=True
        )(x)
        x = Dropout(0.25)(x)

        x = TimeDistributed(Flatten())(x)
        x = LSTM(64, return_sequences=False)(x)
        x = BatchNormalization()(x)
        return x

    def feature_extractor_meteo(input_layer):
        x = LSTM(128, return_sequences=True)(input_layer)
        x = Dropout(0.25)(x)

        x = LSTM(64, return_sequences=True)(x)
        x = Dropout(0.25)(x)

        attention = Attention()([x, x])
        x = LSTM(64, return_sequences=False)(attention)
        x = BatchNormalization()(x)
        return x

    cloud_features = feature_extractor_images(cloud_input)
    meteo_features = feature_extractor_meteo(meteo_input)

    combined_features = Concatenate()([meteo_features, cloud_features])

    x = Dense(128, activation="relu")(combined_features)
    x = Dropout(0.25)(x)

    x = Dense(64, activation="relu")(x)
    x = Dropout(0.25)(x)

    output = Dense(num_classes, activation="softmax", name="output")(x)

    model = Model(inputs=[cloud_input, meteo_input], outputs=output)
    return model


########################
# Load Data and Models #
########################
def load_data(window_sizes=["24", "48", "72"]):
    data_dir = "processed_data_pickles"
    data_loaded = {}
    for w in window_sizes:
        file_path = os.path.join(data_dir, f"data_{w}_hour.pkl")
        with open(file_path, "rb") as f:
            data_loaded[w] = pickle.load(f)
    return data_loaded


def get_data_generators():
    data_loaded = load_data()
    (
        X_train_24,
        y_train_24,
        X_val_24,
        y_val_24,
        X_train_images_24,
        X_val_images_24,
    ) = data_loaded["24"]
    (
        X_train_48,
        y_train_48,
        X_val_48,
        y_val_48,
        X_train_images_48,
        X_val_images_48,
    ) = data_loaded["48"]
    (
        X_train_72,
        y_train_72,
        X_val_72,
        y_val_72,
        X_train_images_72,
        X_val_images_72,
    ) = data_loaded["72"]

    val_generator_24 = HybridDataGenerator(
        X_val_24, X_val_images_24, y_val_24, 32, class_weights=None, shuffle=False
    )
    val_generator_48 = HybridDataGenerator(
        X_val_48, X_val_images_48, y_val_48, 32, class_weights=None, shuffle=False
    )
    val_generator_72 = HybridDataGenerator(
        X_val_72, X_val_images_72, y_val_72, 32, class_weights=None, shuffle=False
    )

    return {24: val_generator_24, 48: val_generator_48, 72: val_generator_72}


def load_model(window_size, weight_type):
    model = hybrid_model_optimized(window_size, num_classes=4)
    optimizer = Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )

    # Load weights
    if window_size == 24:
        if weight_type == "automatic":
            weights_path = "24_automatic_weights/24_window_hybrid_model_automatic_weights_epoch_10.h5"
        else:
            weights_path = "24_manual_weights/24_window_hybrid_model_manual_weights_epoch_10_additional10.keras"
    elif window_size == 48:
        if weight_type == "automatic":
            weights_path = "48_automatic_weights/48_window_hybrid_model_automatic_weights_epoch_20.keras"
        else:
            weights_path = "48_manual_weights/48_window_hybrid_model_manual_weights_epoch_20_additional10.keras"
    else:
        if weight_type == "automatic":
            weights_path = "72_automatic_weights/72_window_hybrid_model_automatic_weights_additional_epoch_10.h5"
        else:
            weights_path = "72_manual_weights/72_window_hybrid_model_manual_weights_additional_epoch_10.h5"

    model.load_weights(weights_path)
    return model


########################
# Histories (hard-coded)
########################
history_24_auto = {
    "accuracy": [
        0.3006,
        0.3854,
        0.4562,
        0.5168,
        0.5726,
        0.6254,
        0.6482,
        0.6600,
        0.6773,
        0.6907,
    ],
    "val_accuracy": [
        0.3353,
        0.3770,
        0.4067,
        0.4226,
        0.4206,
        0.4762,
        0.4722,
        0.4464,
        0.4623,
        0.4702,
    ],
    "loss": [
        1.4186,
        1.3193,
        1.2189,
        1.1311,
        1.0571,
        1.0115,
        0.9468,
        0.9102,
        0.8274,
        0.7956,
    ],
    "val_loss": [
        1.4139,
        1.3756,
        1.2778,
        1.2910,
        1.3577,
        1.2876,
        1.4037,
        1.4152,
        1.4166,
        1.4855,
    ],
}
history_24_manual = {
    "accuracy": [
        0.1538,
        0.2420,
        0.2649,
        0.2664,
        0.3275,
        0.3859,
        0.4302,
        0.4450,
        0.4941,
        0.4969,
    ],
    "val_accuracy": [
        0.2679,
        0.2798,
        0.3056,
        0.3651,
        0.4206,
        0.4345,
        0.4306,
        0.4345,
        0.4405,
        0.4563,
    ],
    "loss": [
        1.9726,
        1.7410,
        1.7041,
        1.5766,
        1.5181,
        1.4653,
        1.3461,
        1.3466,
        1.2249,
        1.2479,
    ],
    "val_loss": [
        1.4832,
        1.4187,
        1.3785,
        1.3800,
        1.3417,
        1.3122,
        1.2941,
        1.3063,
        1.3224,
        1.4047,
    ],
}

history_48_auto = {
    "accuracy": [
        0.5951,
        0.6148,
        0.6099,
        0.6328,
        0.6566,
        0.6391,
        0.6528,
        0.6498,
        0.6599,
        0.6584,
        0.6755,
    ],
    "val_accuracy": [
        0.4702,
        0.4583,
        0.4266,
        0.4683,
        0.4484,
        0.4960,
        0.4901,
        0.4861,
        0.5179,
        0.5020,
        0.4901,
    ],
    "loss": [
        0.9992,
        0.9401,
        0.9730,
        0.9488,
        0.8645,
        0.8934,
        0.8274,
        0.8338,
        0.8217,
        0.8238,
        0.7959,
    ],
    "val_loss": [
        1.2397,
        1.2545,
        1.3380,
        1.3148,
        1.2744,
        1.3512,
        1.3683,
        1.3856,
        1.4162,
        1.4356,
        1.4468,
    ],
}
history_48_manual = {
    "accuracy": [
        0.5734,
        0.5763,
        0.5990,
        0.5961,
        0.6108,
        0.6245,
        0.6288,
        0.6285,
        0.6410,
        0.6422,
    ],
    "val_accuracy": [
        0.4742,
        0.4722,
        0.4742,
        0.4762,
        0.4742,
        0.4841,
        0.5040,
        0.5020,
        0.5060,
        0.5040,
    ],
    "loss": [
        1.1782,
        1.2139,
        1.1578,
        1.1053,
        1.1185,
        1.0790,
        1.0258,
        1.0143,
        0.9710,
        0.9672,
    ],
    "val_loss": [
        1.3054,
        1.2946,
        1.2867,
        1.3817,
        1.4027,
        1.3963,
        1.3601,
        1.3954,
        1.4998,
        1.5691,
    ],
}

history_72_auto = {
    "accuracy": [
        0.3180,
        0.3253,
        0.3772,
        0.4322,
        0.4720,
        0.5104,
        0.5556,
        0.5742,
        0.5840,
        0.5994,
        0.6123,
        0.6162,
        0.6204,
        0.6332,
        0.6319,
        0.6311,
        0.6489,
        0.6589,
        0.6599,
        0.6584,
    ],
    "val_accuracy": [
        0.3194,
        0.3452,
        0.3909,
        0.3929,
        0.4087,
        0.3512,
        0.4603,
        0.3770,
        0.4187,
        0.4782,
        0.4901,
        0.4444,
        0.4921,
        0.4980,
        0.4940,
        0.5159,
        0.5020,
        0.5040,
        0.5000,
        0.4980,
    ],
    "loss": [
        1.4765,
        1.3715,
        1.3360,
        1.2776,
        1.2458,
        1.2021,
        1.1494,
        1.1142,
        1.0757,
        1.0489,
        1.0192,
        1.0031,
        0.9673,
        0.9344,
        0.9389,
        0.9185,
        0.8624,
        0.8492,
        0.8279,
        0.8395,
    ],
    "val_loss": [
        1.3746,
        1.3302,
        1.3127,
        1.3169,
        1.2994,
        1.3915,
        1.2238,
        1.5778,
        1.4420,
        1.2603,
        1.1800,
        1.2640,
        1.2290,
        1.2411,
        1.2527,
        1.3276,
        1.3023,
        1.4191,
        1.3968,
        1.3885,
    ],
}
history_72_manual = {
    "accuracy": [
        0.1469,
        0.1677,
        0.2032,
        0.2806,
        0.3487,
        0.4147,
        0.4876,
        0.5106,
        0.5261,
        0.5179,
        0.5591,
        0.5741,
        0.5738,
        0.5750,
        0.5919,
        0.6003,
        0.6017,
        0.6154,
        0.6266,
        0.6312,
    ],
    "val_accuracy": [
        0.3254,
        0.2956,
        0.4028,
        0.4425,
        0.4127,
        0.4365,
        0.4464,
        0.4583,
        0.4504,
        0.4563,
        0.4960,
        0.4722,
        0.4266,
        0.4702,
        0.4702,
        0.4821,
        0.5079,
        0.5000,
        0.4960,
        0.5079,
    ],
    "loss": [
        1.7466,
        1.6318,
        1.5455,
        1.4819,
        1.4099,
        1.3626,
        1.3071,
        1.2729,
        1.2485,
        1.2407,
        1.1751,
        1.1393,
        1.1605,
        1.1261,
        1.0844,
        1.0593,
        1.0299,
        0.9766,
        0.9537,
        0.9416,
    ],
    "val_loss": [
        1.3736,
        1.4518,
        1.2784,
        1.2894,
        1.2983,
        1.2555,
        1.2536,
        1.2611,
        1.2813,
        1.2927,
        1.3108,
        1.2438,
        1.4202,
        1.3033,
        1.3139,
        1.3453,
        1.4046,
        1.3977,
        1.4109,
        1.4615,
    ],
}

histories = {
    24: {"automatic": history_24_auto, "manual": history_24_manual},
    48: {"automatic": history_48_auto, "manual": history_48_manual},
    72: {"automatic": history_72_auto, "manual": history_72_manual},
}

class_names = ["High Rain", "Less Rain", "Medium Rain", "No Rain"]


########################
# Helper Functions     #
########################
def evaluate_model(model, generator):
    y_pred_probs = model.predict(generator)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = []
    for i in range(len(generator)):
        _, labels, _ = generator[i]
        y_true.append(np.argmax(labels, axis=1))
    y_true = np.concatenate(y_true)
    cm = confusion_matrix(y_true, y_pred)
    # Generate text report (not dict)
    report_text = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=False
    )
    # For other uses, if needed:
    # report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    return y_true, y_pred, y_pred_probs, cm, report_text


def plot_history(history_dict):
    epochs = range(1, len(history_dict["accuracy"]) + 1)
    df = pd.DataFrame(
        {
            "epoch": epochs,
            "accuracy": history_dict["accuracy"],
            "val_accuracy": history_dict["val_accuracy"],
            "loss": history_dict["loss"],
            "val_loss": history_dict["val_loss"],
        }
    )
    acc_chart = (
        alt.Chart(df)
        .transform_fold(fold=["accuracy", "val_accuracy"], as_=["metric", "value"])
        .mark_line(point=True)
        .encode(
            x="epoch:Q",
            y="value:Q",
            color="metric:N",
            tooltip=[
                alt.Tooltip("epoch:Q"),
                alt.Tooltip("value:Q"),
                alt.Tooltip("metric:N"),
            ],
        )
        .properties(title="Accuracy Over Epochs", width=300, height=350)
    )

    loss_chart = (
        alt.Chart(df)
        .transform_fold(fold=["loss", "val_loss"], as_=["metric", "value"])
        .mark_line(point=True)
        .encode(
            x="epoch:Q",
            y="value:Q",
            color="metric:N",
            tooltip=[
                alt.Tooltip("epoch:Q"),
                alt.Tooltip("value:Q"),
                alt.Tooltip("metric:N"),
            ],
        )
        .properties(title="Loss Over Epochs", width=300, height=350)
    )

    return acc_chart, loss_chart


def plot_confusion_matrix(cm):
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    df_melt = df_cm.reset_index().melt("index")
    cm_chart = (
        alt.Chart(df_melt)
        .mark_rect()
        .encode(
            x=alt.X("variable:N", title="Predicted"),
            y=alt.Y("index:N", title="True"),
            color=alt.Color("value:Q", scale=alt.Scale(scheme="blues")),
            tooltip=[
                alt.Tooltip("index:N"),
                alt.Tooltip("variable:N"),
                alt.Tooltip("value:Q"),
            ],
        )
        .properties(title="Confusion Matrix", width=1000, height=800)
    )
    text = cm_chart.mark_text(baseline="middle", fontSize=15).encode(
        text="value:Q",
        color=alt.condition(
            alt.datum.value > (cm.max() / 2), alt.value("white"), alt.value("black")
        ),
    )
    return cm_chart + text


def display_sample(generator, y_pred_probs, sample_class_index=0):
    for i in range(len(generator)):
        inputs, labels, _ = generator[i]
        batch_indices = np.where(np.argmax(labels, axis=1) == sample_class_index)[0]
        if len(batch_indices) > 0:
            sample_index = batch_indices[0]
            global_index = i * generator.batch_size + sample_index
            break
    else:
        st.write(
            f"No samples found for class index {sample_class_index} ({class_names[sample_class_index]})."
        )
        return

    sample_meteo = inputs["meteo_input"][sample_index]
    sample_images = inputs["cloud_input"][sample_index]
    true_label = class_names[np.argmax(labels[sample_index])]
    predicted_probs = y_pred_probs[global_index]
    predicted_label = class_names[np.argmax(predicted_probs)]

    # Show input data and true label first
    st.write("### Sample Input")
    st.write("**Meteorological Data (Sample Input):**")
    meteo_features = [f"Feature {i}" for i in range(sample_meteo.shape[1])]
    meteo_df = pd.DataFrame(sample_meteo, columns=meteo_features)
    st.dataframe(meteo_df)

    st.write("**Cloud Images (Sample Input):**")
    num_images = sample_images.shape[0]
    cols = st.columns(min(num_images, 6))
    for idx, img_frame in enumerate(sample_images):
        img = (img_frame.squeeze() * 255).astype(np.uint8)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        col = cols[idx % len(cols)]
        col.image(img_rgb, caption=f"Image {idx+1}", use_column_width=True)

    st.write(f"**True Label:** {true_label}")

    # Now show model prediction
    st.write("### Model Prediction")
    color = "green" if predicted_label == true_label else "red"
    st.markdown(
        f"**Predicted Label:** <span style='color:{color}'>{predicted_label}</span>",
        unsafe_allow_html=True,
    )

    st.write("**Predicted Probabilities:**")
    prob_df = pd.DataFrame(predicted_probs, index=class_names, columns=["Probability"])
    st.table(prob_df)


########################
# Streamlit Interface  #
########################

st.title("ConvLSTM + LSTM (with Attention) Model Performance & Results")

window_size = st.sidebar.selectbox("Select Forecast Horizon (Hours):", [24, 48, 72])

# Only show weight type radio after window size selected
if window_size:
    weight_type = st.sidebar.radio(
        "Select Weighting Strategy:", ["automatic", "manual"]
    )
else:
    weight_type = None

# Show a button to run model after both are selected
if window_size and weight_type:
    run_button = st.sidebar.button("Run Model")
else:
    run_button = False

if run_button:
    generators = get_data_generators()
    model = load_model(window_size, weight_type)
    history_dict = histories[window_size][weight_type]

    # Evaluate model
    y_true, y_pred, y_pred_probs, cm, report_text = evaluate_model(
        model, generators[window_size]
    )

    # Plot training history
    acc_chart, loss_chart = plot_history(history_dict)
    st.altair_chart(acc_chart, use_container_width=True)
    st.altair_chart(loss_chart, use_container_width=True)

    # Show classification report as text (like console print)
    st.write("### Classification Report")
    st.text(report_text)  # Just print the text version

    # Show confusion matrix
    st.write("### Confusion Matrix")
    cm_chart = plot_confusion_matrix(cm)
    st.altair_chart(cm_chart, use_container_width=True)

    # Display sample
    display_sample(generators[window_size], y_pred_probs, sample_class_index=0)
