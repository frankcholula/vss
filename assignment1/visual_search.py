import os
import random
import time
import logging
import streamlit as st
from data_sources import FirebaseConnection
from descriptor import Descriptor
from retriever import Retriever
from ground_truth import ImageLabeler
from metrics import ClassBasedEvaluator, LabelBasedEvaluator
logging.basicConfig(level=logging.INFO)

@st.cache_resource(show_spinner=False)
def load_data():
    firebase_conn = FirebaseConnection()
    bucket = firebase_conn.get_bucket()    
    image_directory = "MSRC_ObjCategImageDatabase_v2/Images"
    local_image_dir = "MSRC_ObjCategImageDatabase_v2_local/Images"
    # TODO: load ground truth labels
    required_file_count = 591
    sleep_time = 1.5
    message, success = firebase_conn.check_local_dir(local_image_dir, required_file_count)
    if success:
        time.sleep(sleep_time)
        message.empty()
        return
    else:
        os.makedirs(local_image_dir, exist_ok=True)
        blobs = list(bucket.list_blobs(prefix=image_directory))
        status = firebase_conn.download_images(blobs, local_image_dir, max_download=required_file_count)
        time.sleep(sleep_time)
        status.empty()
    time.sleep(sleep_time)
    message.empty()


class SessionStateManager:
    def __init__(self, image_files):
        self.image_files = image_files
        self.init_session_state()

    def init_session_state(self):
        if 'bins' not in st.session_state:
            st.session_state['bins'] = 32
        if 'selected_image' not in st.session_state:
            st.session_state['selected_image'] = self.image_files[0]
        if 'quant_lvl' not in st.session_state:
            st.session_state['quant_lvl'] = 8
        if 'metric' not in st.session_state:
            st.session_state['metric'] = "l2"
        if 'recompute' not in st.session_state:
            st.session_state['recompute'] = True
        if 'descriptor' not in st.session_state:
            st.session_state['descriptor'] = "globalRGBhisto_quant"
        if 'result_num' not in st.session_state:
            st.session_state['result_num'] = 5
        if 'grid_size' not in st.session_state:
            st.session_state['grid_size'] = 4

    def update_metric(self):
        st.session_state['metric'] = st.session_state['metric_radio']

    def update_bins(self):
        if st.session_state['bins'] != st.session_state['bins_slider']:
            st.session_state['bins'] = st.session_state['bins_slider']
            st.session_state['recompute'] = True
        else:
            self.update_recompute(False)
    
    def update_quant(self):
        if st.session_state['quant_lvl'] != st.session_state['quant_slider']:
            st.session_state['quant_lvl'] = st.session_state['quant_slider']
            st.session_state['recompute'] = True
        else:
            self.update_recompute(False)
    
    def update_descriptor(self):
        if st.session_state['descriptor'] != st.session_state['descriptor_selectbox']:
            logging.info(f"Updating descriptor to {st.session_state['descriptor_selectbox']}")
            st.session_state['descriptor'] = st.session_state['descriptor_selectbox']
            # TODO: fix the path here
            if os.path.exists(f"descriptors/{st.session_state['descriptor']}"):
                self.update_recompute(False)
            else:
                self.update_recompute(True)
        else:
            self.update_recompute(False)

    def update_result_num(self):
        st.session_state['result_num'] = st.session_state['result_num_slider']

    def update_grid_size(self):
        st.session_state['grid_size'] = st.session_state['grid_slider']
        st.session_state['recompute'] = True
    
    def update_recompute(self, recompute:bool):
        st.session_state['recompute'] = recompute


def main():
    load_data()
    DATASET_FOLDER = "MSRC_ObjCategImageDatabase_v2_local"
    DESCRIPTOR_FOLDER = "descriptors"
    image_files = [f for f in os.listdir(os.path.join(DATASET_FOLDER, 'Images')) if f.endswith('.bmp')]
    session_manager = SessionStateManager(image_files)
    labeler = ImageLabeler(DATASET_FOLDER)

    # Section to choose the image and the descriptor

    st.title("Visual Search Engine 👀")
    st.toggle(
        "Debug",
        key="debug_mode",
        help="Toggle to display the ground truth labels for the images."
    )
    cols = st.columns([1.75,1.75,1])
    selected_image = cols[0].selectbox(
        "Choose an Image...",
        image_files,
        index=image_files.index(st.session_state['selected_image']))
    

    # TODO: Add new descriptor options here
    descriptor_method = cols[1].selectbox(
        "Choose your Descriptor...",
        options=['gridRGB','globalRGBhisto_quant','globalRGBhisto', 'rgb', 'random'],
        key="descriptor_selectbox",
        on_change=session_manager.update_descriptor,
        )
    
    match descriptor_method:
        case "globalRGBhisto":
            cols[1].select_slider(
                "Select the Number of Bins...",
                options = [8, 16, 32, 64, 128, 256],
                value=st.session_state['bins'],
                key="bins_slider",
                on_change=session_manager.update_bins
            )
        case "globalRGBhisto_quant":
            cols[1].select_slider(
                label = "Select Your Quantization Level...",
                options = [4, 8, 16, 32],
                help="The number of quantization levels ranges from coarse to fine.",
                value=st.session_state['quant_lvl'],
                key="quant_slider",
                on_change=session_manager.update_quant
            )
        case "gridRGB":
            cols[1].select_slider(
                label = "Select Your Grid Size...",
                options = [2, 4, 8, 16],
                help="Determines how the image is divided horizontally and vertically. ",
                value=st.session_state['grid_size'],
                key="grid_slider",
                on_change=session_manager.update_grid_size
            )
    
    descriptor = Descriptor(
        DATASET_FOLDER,
        DESCRIPTOR_FOLDER,
        descriptor_method,
        bins=st.session_state['bins'],
        quant_lvl=st.session_state['quant_lvl'],
        grid_size=st.session_state['grid_size']
    )
    if st.session_state['recompute']:
        logging.info("Recomputing descriptors...")
        descriptor.extract(st.session_state['recompute'])
        session_manager.update_recompute(False)

    # Debug the descriptors here
    img2descriptors = descriptor.get_image_descriptor_mapping()

    # Button to select a random image
    cols[2].markdown("<div style='width: 1px; height: 28px'></div>", unsafe_allow_html=True)
    if cols[2].button("I'm Feeling Lucky"):
        st.session_state['selected_image']  =  random.choice(image_files)
        selected_image = st.session_state['selected_image']
        # need rerun here to refresh selected image value
        st.rerun()
    
    metric = cols[2].radio(
        "Comparison Metric",
        options=["l2", "l1"],
        index=["l2", "l1"].index(st.session_state['metric']),
        key="metric_radio",
        on_change=session_manager.update_metric
    )

    result_num = cols[0].slider(
        "Number of Similar Images to Retrieve...",
        min_value=5,
        max_value=30,
        value=st.session_state.get('result_num', 5),
        step=1,
        format="%d",
        key="result_num_slider",
        on_change=session_manager.update_result_num
    )
    
    # Section to display the query image and the top similar images
    left_col, right_col = st.columns([2.25, 2.25])
    with left_col:
        st.header("Query Image:")
        st.image(os.path.join(DATASET_FOLDER, 'Images', selected_image), use_column_width=True)
        if st.session_state['debug_mode']:
            st.write(f"Class: {labeler.get_class(selected_image)}")
            st.write(labeler.get_labels(selected_image))
    

    retriever = Retriever(img2descriptors, metric)
    similar_images = retriever.retrieve(os.path.join(DATASET_FOLDER, 'Images', selected_image), number=result_num)
    
    with right_col:
        st.header("Ground Truth:")
        gt_img = labeler.load_img(selected_image)
        st.image(gt_img, use_column_width=True)
        if st.session_state['debug_mode']:
            st.write(f"Class: {labeler.get_class(selected_image)}")
            st.write(labeler.get_labels(selected_image))

    st.header(f"Top {result_num} Similar Images:")
    for i in range(0, len(similar_images), 5):
        cols = st.columns(5)
        for col, img_path in zip(cols, similar_images[i:i+5]):
            col.image(img_path, use_column_width=True, caption=os.path.basename(img_path))
            if st.session_state['debug_mode']:
                col.write(f"Class: {labeler.get_class(os.path.basename(img_path))}")
                col.write(labeler.get_labels(os.path.basename(img_path)))

    tab1, tab2 = st.tabs(["Class-based Performance", "Label-based Performance"])
    with tab1:
        input_class = labeler.get_class(selected_image)
        retrieved_image_classes = [labeler.get_class(os.path.basename(img_path)) for img_path in similar_images]
        cbe = ClassBasedEvaluator(input_class, retrieved_image_classes)
        cm = cbe.create_class_matrix(input_class, retrieved_image_classes)
        cbe.plot_class_matrix(cm, input_class)

        all_labels = labeler.get_all_labels()
        total_relevant = sum(1 for image_data in all_labels.values() if image_data['class'] == input_class)
        precisions, recalls = cbe.calculate_pr_curve(input_class, retrieved_image_classes, total_relevant)
        cbe.plot_pr_curve(precisions, recalls)
    with tab2:
        input_class_labels = labeler.get_labels(selected_image)
        retrieved_image_labels = [labeler.get_labels(os.path.basename(img_path)) for img_path in similar_images]
        lbe = LabelBasedEvaluator(input_class_labels, retrieved_image_labels)
        labels_matrix = lbe.create_labels_matrix(input_class_labels, retrieved_image_labels)
        lbe.plot_labels_matrix(labels_matrix)

    

if __name__ == "__main__":
    main()