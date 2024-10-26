import cv2
import os
import random
import time
import logging
import streamlit as st
from data_sources import FirebaseConnection
from descriptors import Descriptor
from retrievers import Retriever
from ground_truth import ImageLabeler
from metrics import ClassBasedEvaluator, LabelBasedEvaluator
from session_state_managers import SessionStateManager
from feature_detectors import FeatureDetector
from sift_visualizer import visualize_sift


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


@st.cache_resource(show_spinner=False)
def load_data():
    firebase_conn = FirebaseConnection()
    bucket = firebase_conn.get_bucket()
    image_directory = "MSRC_ObjCategImageDatabase_v2/Images"
    local_image_dir = "MSRC_ObjCategImageDatabase_v2_local/Images"
    # TODO: load ground truth labels
    required_file_count = 591
    sleep_time = 1.5
    message, success = firebase_conn.check_local_dir(
        local_image_dir, required_file_count
    )
    if success:
        time.sleep(sleep_time)
        message.empty()
        return
    else:
        os.makedirs(local_image_dir, exist_ok=True)
        blobs = list(bucket.list_blobs(prefix=image_directory))
        status = firebase_conn.download_images(
            blobs, local_image_dir, max_download=required_file_count
        )
        time.sleep(sleep_time)
        status.empty()
    time.sleep(sleep_time)
    message.empty()


def main():
    load_data()
    DATASET_FOLDER = "MSRC_ObjCategImageDatabase_v2_local"
    DESCRIPTOR_FOLDER = "descriptors"
    image_files = [
        f
        for f in os.listdir(os.path.join(DATASET_FOLDER, "Images"))
        if f.endswith(".bmp")
    ]
    session_manager = SessionStateManager(image_files)
    labeler = ImageLabeler(DATASET_FOLDER)

    # Section to choose the image and the descriptor
    vse, sv = st.tabs(["Visual Search Engine", "SIFT Visualizer"])
    vse.title("Visual Search Engine 👀")

    header_cols = vse.columns([3, 3, 3, 2])
    header_cols[3].markdown(
        "<div style='width: 1px; height: 28px'></div>", unsafe_allow_html=True
    )
    header_cols[3].toggle(
        "Debug",
        key="debug_mode",
        help="Toggle to display the ground truth labels for the images.",
    )
    with vse.expander("**Expand to tweak hyper-parameters!**", icon="🎛️"):
        option_cols = st.columns([3, 3, 2])

    selected_image = header_cols[0].selectbox(
        "**🖼️ Choose an Image...**",
        image_files,
        index=image_files.index(st.session_state["selected_image"]),
    )

    # TODO: Add new descriptor options here
    descriptor_method = header_cols[1].selectbox(
        "**🎨 Choose a Descriptor...**",
        options=[
            "SIFT",
            "gridCombined",
            "gridEOhisto",
            "gridRGB",
            "globalRGBhisto_quant",
            "globalRGBhisto",
            "rgb",
            "random",
        ],
        key="descriptor_selectbox",
        on_change=session_manager.update_descriptor,
    )

    match descriptor_method:
        case "globalRGBhisto":
            option_cols[1].select_slider(
                "Select the Number of Bins...",
                options=[8, 16, 32, 64, 128, 256],
                value=st.session_state["bins"],
                key="bins_slider",
                on_change=session_manager.update_bins,
            )
        case "globalRGBhisto_quant":
            option_cols[1].select_slider(
                label="Select Your Quantization Level...",
                options=[4, 8, 16, 32],
                help="The number of quantization levels ranges from coarse to fine.",
                value=st.session_state["quant_lvl"],
                key="quant_slider",
                on_change=session_manager.update_quant,
            )
        case "gridRGB":
            option_cols[1].select_slider(
                label="Select Your Grid Size...",
                options=[2, 4, 8, 16],
                help="Determines how the image is divided horizontally and vertically. ",
                value=st.session_state["grid_size"],
                key="grid_slider",
                on_change=session_manager.update_grid_size,
            )
        case "gridEOhisto":
            option_cols[1].select_slider(
                label="Select Your Grid Size...",
                options=[2, 4, 8, 16],
                help="Determines how the image is divided horizontally and vertically. ",
                value=st.session_state["grid_size"],
                key="grid_slider",
                on_change=session_manager.update_grid_size,
            )
            option_cols[1].select_slider(
                label="Select Your Sobel Filter Size...",
                options=[3, 5, 7],
                help="Determines the size of the Sobel filter.",
                value=st.session_state["sobel_filter_size"],
                key="sobel_filter_slider",
                on_change=session_manager.update_sobel_filter_size,
            )
            option_cols[1].select_slider(
                "Select the Angular Quantization Level...",
                options=range(8, 33),
                value=st.session_state["ang_quant_lvl"],
                key="ang_quant_slider",
                on_change=session_manager.update_ang_quant_lvl,
            )
        case "gridCombined":
            option_cols[1].select_slider(
                label="Select Your Grid Size...",
                options=[2, 4, 8, 16],
                help="The smaller the grid size, the more broad, global features it captures.",
                value=st.session_state["grid_size"],
                key="grid_slider",
                on_change=session_manager.update_grid_size,
            )
            option_cols[1].select_slider(
                label="Select Your Sobel Filter Size...",
                options=[3, 5, 7],
                help="The larger the kernel, the coarser the edge and less sensitive to noise.",
                value=st.session_state["sobel_filter_size"],
                key="sobel_filter_slider",
                on_change=session_manager.update_sobel_filter_size,
            )
            option_cols[1].select_slider(
                "Select the Angular Quantization Level...",
                options=range(8, 33),
                value=st.session_state["ang_quant_lvl"],
                help="The higher the quantizatino, the finder the edge orientation histogram, but also more prone to noise.",
                key="ang_quant_slider",
                on_change=session_manager.update_ang_quant_lvl,
            )
            option_cols[2].radio(
                "Normalization Method",
                options=["minmax", "zscore"],
                help="Used to normalize EO histograms and RGB histograms.",
                key="norm_method_radio",
                on_change=session_manager.update_norm_method,
            )
        case "SIFT":
            pass

    # TODO: Add new descriptor options here
    descriptor = Descriptor(
        DATASET_FOLDER,
        DESCRIPTOR_FOLDER,
        descriptor_method,
        bins=st.session_state["bins"],
        quant_lvl=st.session_state["quant_lvl"],
        grid_size=st.session_state["grid_size"],
        sobel_filter_size=st.session_state["sobel_filter_size"],
        ang_quant_lvl=st.session_state["ang_quant_lvl"],
        norm_method=st.session_state["norm_method"],
        # TODO: add feature detection method here
        feature_detector="SIFT",
    )
    if st.session_state["recompute"]:
        LOGGER.info("Recomputing descriptors...")
        descriptor.extract(st.session_state["recompute"])
        session_manager.update_recompute(False)

    # Debug the descriptors here
    img2descriptors = descriptor.get_image_descriptor_mapping()

    # Button to select a random image
    header_cols[2].markdown(
        "<div style='width: 1px; height: 28px'></div>", unsafe_allow_html=True
    )
    if header_cols[2].button("🎲 I'm Feeling Lucky"):
        st.session_state["selected_image"] = random.choice(image_files)
        selected_image = st.session_state["selected_image"]
        # need rerun here to refresh selected image value
        st.rerun()

    metric = option_cols[2].radio(
        "Comparison Metric",
        options=["L2", "L1", "Mahalanobis"],
        index=["L2", "L1", "Mahalanobis"].index(st.session_state["metric"]),
        key="metric_radio",
        on_change=session_manager.update_metric,
    )

    result_num = option_cols[0].slider(
        "Number of Images to Retrieve...",
        min_value=5,
        max_value=30,
        value=st.session_state.get("result_num", 5),
        step=1,
        format="%d",
        key="result_num_slider",
        on_change=session_manager.update_result_num,
    )

    # Section to display the query image and the top similar images
    left_col, right_col = vse.columns([1, 1])
    with left_col:
        st.header("Query")
        st.image(
            os.path.join(DATASET_FOLDER, "Images", selected_image),
            use_column_width=True,
        )
        if st.session_state["debug_mode"]:
            st.write(f"Class: {labeler.get_class(selected_image)}")
            st.write(labeler.get_labels(selected_image))

    retriever = Retriever(img2descriptors, metric)
    similar_images = retriever.retrieve(
        os.path.join(DATASET_FOLDER, "Images", selected_image), number=result_num
    )

    with right_col:
        st.header("Ground Truth")
        gt_img = labeler.load_img(selected_image)
        st.image(gt_img, use_column_width=True)
        if st.session_state["debug_mode"]:
            st.write(f"Class: {labeler.get_class(selected_image)}")
            st.write(labeler.get_labels(selected_image))

    vse.header(f"Top {result_num} Similar Images")
    for i in range(0, len(similar_images), 5):
        cols = vse.columns(5)
        for col, img_path in zip(cols, similar_images[i : i + 5]):
            col.image(
                img_path, use_column_width=True, caption=os.path.basename(img_path)
            )
            if st.session_state["debug_mode"]:
                col.write(f"Class: {labeler.get_class(os.path.basename(img_path))}")
                col.write(labeler.get_labels(os.path.basename(img_path)))
    tab1, tab2 = vse.tabs(["Class-based Performance", "Label-based Performance"])
    good_class_based = False
    good_label_based = False
    with tab1:
        input_class = labeler.get_class(selected_image)
        labels_dict = labeler.get_labels_dict()
        retrieved_image_classes = [
            labeler.get_class(os.path.basename(img_path)) for img_path in similar_images
        ]
        cbe = ClassBasedEvaluator(input_class, retrieved_image_classes)
        cm = cbe.create_class_matrix(input_class, retrieved_image_classes)
        tri = cbe.count_total_relevant_images(selected_image, labels_dict)
        min_tri = min(tri, result_num)
        fetched = cm[input_class].iloc[0]

        st.write(
            f"**In the top `{result_num}` results, you retrieved `{fetched}` images in `class {input_class}`. There are `{tri}` total relevant images.**"
        )
        if fetched == result_num:
            st.toast("Good class-based performance!", icon="😍")
            time.sleep(0.5)
            good_class_based = True

        cbe.plot_class_matrix(cm, input_class)
        precisions, recalls, f1_scores = cbe.calculate_pr_f1_values(
            input_class, retrieved_image_classes, min_tri
        )
        cbe.plot_pr_curve(precisions, recalls)
        cbe.plot_f1_score(f1_scores)
    with tab2:
        input_class_labels = labeler.get_labels(selected_image)
        retrieved_image_labels = [
            labeler.get_labels(os.path.basename(img_path))
            for img_path in similar_images
        ]
        lbe = LabelBasedEvaluator(input_class_labels, retrieved_image_labels)
        lm = lbe.create_labels_matrix()
        tri = lbe.count_total_relevant_images(selected_image, labeler.get_labels_dict())
        fetched = (lm == 1).any().sum()
        min_tri = min(tri, result_num)
        st.write(
            f"**In the top `{min_tri}` results, you retrieved `{fetched}` with one of these labels:`{input_class_labels}`.**"
        )
        st.write(f"**There are `{tri}` total relevant images.**")
        if fetched == result_num:
            st.toast("Good label-based performance!", icon="😍")
            time.sleep(0.5)
            good_label_based = True
        lbe.plot_labels_matrix(lm)
        lbe.plot_pr_curve(min_tri)
        lbe.plot_f1_score(min_tri)
    if good_class_based and good_label_based:
        st.balloons()

    sv.title("SIFT Visualizer 🪄")
    sv_select_image = sv.selectbox(
        "**🖼️ Choose an Image...**",
        image_files,
        key="sv_select_image",
    )
    sv_left_col, sv_right_col = sv.columns([1, 1])
    with sv_left_col:
        query_img = cv2.imread(os.path.join(DATASET_FOLDER, "Images", sv_select_image))
        st.header("Query Image")
        st.image(cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB), use_column_width=True)
        st.header("Greyscale")
        st.image(cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY), use_column_width=True)

    with sv_right_col:
        fd = FeatureDetector("SIFT")

        selected_img_obj = cv2.imread(
            os.path.join(DATASET_FOLDER, "Images", sv_select_image)
        )
        kp, desc = fd.detect_keypoints_compute_descriptors(
            selected_img_obj,
        )
        img_with_kp = cv2.drawKeypoints(
            selected_img_obj, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        img_with_kp_rgb = cv2.cvtColor(img_with_kp, cv2.COLOR_BGR2RGB)
        st.header("Keypoints")
        st.image(img_with_kp_rgb, use_column_width=True)
        ghost_img_rgb = visualize_sift(selected_img_obj)[0]
        st.header("SIFT")
        st.image(ghost_img_rgb, use_column_width=True)


if __name__ == "__main__":
    main()
