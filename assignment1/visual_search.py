import os
import random
import time
import streamlit as st
from data_sources import FirebaseConnection
from descriptor import Descriptor
from retriever import Retriever
import logging

logging.basicConfig(level=logging.INFO)

@st.cache_resource(show_spinner=False)
def load_data():
    firebase_conn = FirebaseConnection()
    bucket = firebase_conn.get_bucket()    
    # Directory in Firebase storage
    image_directory = "MSRC_ObjCategImageDatabase_v2/Images"
    # Create a local directory to store images
    local_image_dir = "MSRC_ObjCategImageDatabase_v2_local/Images"
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
            st.session_state['descriptor'] = "rgb"


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


    def update_recompute(self, recompute:bool):
        st.session_state['recompute'] = recompute

def main():
    load_data()
    DATASET_FOLDER = "MSRC_ObjCategImageDatabase_v2_local"
    DESCRIPTOR_FOLDER = "descriptors"
    image_files = [f for f in os.listdir(os.path.join(DATASET_FOLDER, 'Images')) if f.endswith('.bmp')]
    session_manager = SessionStateManager(image_files)
    
    # Section to choose the image and the descriptor
    st.title("Visual Search Engine 👀")
    cols = st.columns([1.75,1.75,1])
    selected_image = cols[0].selectbox(
        "Choose an Image...",
        image_files,
        index=image_files.index(st.session_state['selected_image']))
    
    metric = cols[0].radio(
        "Select Comparison Metric...",
        options=["l2", "l1"],
        index=["l2", "l1"].index(st.session_state['metric']),
        key="metric_radio",
        on_change=session_manager.update_metric
    )

    # TODO: Add new descriptor options here
    descriptor_method = cols[1].selectbox(
        "Choose your Descriptor...",
        options=['rgb', 'random', 'globalRGBhisto', 'globalRGBhisto_quant'],
        key="descriptor_selectbox",
        on_change=session_manager.update_descriptor,
    )
    
    if descriptor_method == "globalRGBhisto":
        cols[1].select_slider(
            "Select the Number of Bins...",
            options = [8, 16, 32, 64, 128, 256],
            value=32,
            key="bins_slider",
            on_change=session_manager.update_bins
    )

    if descriptor_method == "globalRGBhisto_quant":
        cols[1].select_slider(
            "Select Your Quantization Level...",
            options = [4,8, 16, 32],
            value=8,
            key="quant_slider",
            on_change=session_manager.update_quant
        )
    
    descriptor = Descriptor(
        DATASET_FOLDER,
        DESCRIPTOR_FOLDER,
        descriptor_method,
        bins=st.session_state['bins'],
        quant_lvl=st.session_state['quant_lvl']
    )
    if st.session_state['recompute']:
        logging.info("Recomputing descriptors...")
        descriptor.extract(st.session_state['recompute'])
        session_manager.update_recompute(False)

    img2descriptors = descriptor.get_image_descriptor_mapping()
    # TODO: debug
    for img, desc in img2descriptors.items():
        print(f"Image: {img}, Descriptor: {desc.shape}")

    # Button to select a random image
    cols[2].markdown("<div style='width: 1px; height: 28px'></div>", unsafe_allow_html=True)
    if cols[2].button("I'm Feeling Lucky"):
        st.session_state['selected_image']  =  random.choice(image_files)
        selected_image = st.session_state['selected_image']
        # need rerun here to refresh selected image value
        st.rerun()
    
    # Section to display the query image and the top similar images
    st.write("Query Image:")
    st.image(os.path.join(DATASET_FOLDER, 'Images', selected_image), use_column_width=True)
    result_num = 10
    retriever = Retriever(img2descriptors, metric)
    similiar_images = retriever.retrieve(os.path.join(DATASET_FOLDER, 'Images', selected_image), number=result_num)
    st.write(f"Top {result_num} similar images:")
    cols = st.columns(result_num)
    for col, img_path in zip(cols, similiar_images):
        col.image(img_path, use_column_width=True)

if __name__ == "__main__":
    main()