import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io




# Add CSS styling
st.markdown("""
    <style>
        /* General styling */
        body {
            background-color: #f4f4f9;
            font-family: 'Arial', sans-serif;
        }

        /* Title and subtitle */
        .title {
            color: #1e3d58;
            font-size: 40px;
            font-weight: bold;
            text-align: center;
        }

        .subtitle {
            color: #3e5c73;
            font-size: 20px;
            font-weight: 500;
            text-align: center;
        }

        /* Sidebar styling */
        .sidebar .sidebar-content {
            background-color: #e0f7fa;
            color: #00796b;
        }

        .sidebar .sidebar-header {
            background-color: #00796b;
            color: white;
        }

        /* Buttons Styling */
        .stButton button {
            background-color: #00796b;
            color: white;
            border-radius: 5px;
            padding: 10px 20px;
            border: none;
            box-shadow: none;
            transition: background-color 0.3s ease, outline 0.3s ease;
        }

        .stButton button:hover {
            background-color: #004d40;
            outline: 2px solid black;  /* Black outline on hover */
        }

        /* Image styling */
        .stImage {
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);  /* subtle shadow for images */
        }

        /* Styling for download buttons */
        .stDownloadButton button {
            background-color: #009688;
            color: white;
            border-radius: 5px;
            padding: 10px 20px;
            transition: background-color 0.3s ease, outline 0.3s ease;
        }

        .stDownloadButton button:hover {
            background-color: #00796b;  /* Darker shade of green for hover */
            outline: 2px solid black;  /* Black outline on hover */
        }

        /* Consistent padding and margins for containers */
        .stColumn {
            padding: 10px;
        }

        /* Section headers */
        .section-header {
            color: #004d40;
            font-size: 25px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 20px;
        }

        /* File uploader styling */
        .stFileUploader {
            background-color: #ffffff;
            border-radius: 5px;
            padding: 20px;
        }

        /* Images grid styling */
        .stImageGrid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
            gap: 20px;
            justify-items: center;
        }

        /* Banner image styling */
        .stBannerImage {
            border-radius: 10px;
            border: 2px solid #004d40;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
    </style>
""", unsafe_allow_html=True)


# Load CycleGAN model weights
@st.cache_resource
def load_cycle_gan_model(weight_file):
    model = tf.keras.models.load_model(weight_file, compile=False)
    return model

# Function to preprocess input image
def preprocess_image(uploaded_image):
    img = Image.open(uploaded_image)
    img = img.resize((256, 256))  # Resize to the expected input size
    img_array = np.array(img)
    img_array = (img_array / 127.5) - 1.0  # Normalize to [-1, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array, img

# Function to postprocess and save translated image
def postprocess_image(prediction):
    prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
    return Image.fromarray(prediction)

# Streamlit App
def main():
    st.title("CycleGAN Style Transfer")
    st.write("Transformasi style warna batik madura dengan gaya bali.")
    
    # Sidebar for selecting the pretrained model
    model_choice = st.sidebar.selectbox(
        "Select Pretrained Model",
        ["Bali Corak Hijau"]
    )
    st.sidebar.write(f"You selected: {model_choice}")
    
    # Mapping the selected model to the correct path and banner image
    model_paths = {
        "Bali Corak Hijau": "cyclegan_checkpoints.040hijau",
        "Bali Corak Coklat Kuning": "cyclegan_checkpoints.040coklat",
        "Bali Corak Merah": "cyclegan_checkpoints.040", 
        "Bali Corak Biru": "cyclegan_checkpoints.040biru"
    }
    banner_images = {
        "Bali Corak Merah": "Bali Merah.png", 
        "Bali Corak Coklat Kuning": "COKLATKUNING.png", 
        "Bali Corak Biru": "madurabiru.png", 
        "Bali Corak Hijau": "balihijau.png"
    }
    
    # Display large banner image for the selected model
    banner_img_path = banner_images.get(model_choice, "default_banner.jpg")
    banner_img = Image.open(banner_img_path)
    banner_img = banner_img.resize((800, 400))  # Resize banner image (adjust as needed)
    st.image(banner_img, caption=f"Style for {model_choice}", use_container_width=True)

    # Sidebar button to show Batik Madura images
    if st.sidebar.button("Show Batik Madura Images"):
        st.write("**Here are some Batik Madura images you can download:**")

        # Batik Madura images directory
        batik_images = {
            "Batik Madura 1": "TEST_MADURA/1 (1).jpg", 
            "Batik Madura 2": "TEST_MADURA/1 (2).jpg", 
            "Batik Madura 3": "TEST_MADURA/1 (3).jpg", 
            "Batik Madura 4": "TEST_MADURA/1 (4).jpg", 
            "Batik Madura 5": "TEST_MADURA/1 (5).jpg", 
            "Batik Madura 6": "TEST_MADURA/1 (6).jpg", 
            "Batik Madura 7": "TEST_MADURA/1 (7).jpg", 
            "Batik Madura 8": "TEST_MADURA/1 (8).jpg", 
            "Batik Madura 9": "TEST_MADURA/1 (9).jpg", 
            "Batik Madura 10": "TEST_MADURA/1 (10).jpg", 
        }

        # Display Batik Madura images in a grid with consistent size
        cols = st.columns(5)  # Create 5 columns for the grid
        for i, (batik_name, batik_path) in enumerate(batik_images.items()):
            # Resize image to a consistent size
            batik_img = Image.open(batik_path).resize((150, 150))  # Resize to 150x150

            # Select the appropriate column
            col = cols[i % 5]
            with col:
                st.image(batik_img, caption=batik_name, use_container_width=True)  # Fixed the error
                # Add download button for each image
                buf = io.BytesIO()
                batik_img.save(buf, format="PNG")
                byte_im = buf.getvalue()
                st.download_button(
                    label="Download",
                    data=byte_im,
                    file_name=f"{batik_name}.png",
                    mime="image/png"
                )

    # Multiple image upload for input images
    uploaded_files = st.file_uploader("Choose images...", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    
    if uploaded_files:
        # Load model and perform predictions for all uploaded images
        weight_file = model_paths.get(model_choice, "path_to_default_model")
        cycle_gan_model = load_cycle_gan_model(weight_file)
        
        # Initialize columns for displaying multiple images
        num_images = len(uploaded_files)
        cols = st.columns(num_images)
        
        for i, uploaded_file in enumerate(uploaded_files):
            img_array, original_img = preprocess_image(uploaded_file)
            
            # Perform prediction
            st.write(f"Processing image {i + 1}...")
            prediction = cycle_gan_model.gen_G(img_array, training=False)[0].numpy()
            
            # Postprocess translated image
            translated_img = postprocess_image(prediction)
            
            # Display images side by side
            with cols[i]:
                st.image(original_img, caption=f"Original Image {i + 1}", width=200)
                st.image(translated_img, caption=f"Translated Image {i + 1}", width=200)
                
                # Option to save translated image
                buf = io.BytesIO()
                translated_img.save(buf, format="PNG")
                byte_im = buf.getvalue()
                st.download_button(
                    label=f"Download Translated Image {i + 1}",
                    data=byte_im,
                    file_name=f"translated_image_{i + 1}.png",
                    mime="image/png"
                )

if __name__ == "__main__":
    main()


    # # Add external link at the bottom
    # st.markdown(
    #     """
    #     ---
    #     **Kunjungi [situs resmi Batik Madura](https://www.contohsitusbatikmadura.com)** 
    #     untuk informasi lebih lanjut tentang desain dan budaya Batik Madura.
    #     """,
    #     unsafe_allow_html=True
    # )

    # Add external link at the bottom with soft and elegant styling
# Add external link at the bottom with soft and elegant styling
st.markdown(
    """
    ---
    <div style="display: flex; justify-content: flex-end; align-items: center; padding: 20px 0; gap: 10px;">
        <a href="https://stylecoklatkuningbatikbali.streamlit.app/" target="_blank" style="text-decoration: none; font-size: 14px; color: #333333; font-weight: 400; background-color: #EAEAEA; padding: 5px 12px; border-radius: 8px; box-shadow: 0px 1px 6px rgba(0, 0, 0, 0.05); display: flex; align-items: center; transition: all 0.3s ease;">
            <img src="https://img.icons8.com/ios-filled/50/333333/link.png" width="15" style="margin-right: 6px;"/> Style Batik Bali Coklat Kuning
        </a>
        <a href="https://stylebirubatikbali.streamlit.app/" target="_blank" style="text-decoration: none; font-size: 14px; color: #333333; font-weight: 400; background-color: #EAEAEA; padding: 5px 12px; border-radius: 8px; box-shadow: 0px 1px 6px rgba(0, 0, 0, 0.05); display: flex; align-items: center; transition: all 0.3s ease;">
            <img src="https://img.icons8.com/ios-filled/50/333333/link.png" width="15" style="margin-right: 6px;"/> Style Batik Bali Biru
        </a>
        <a href="https://cycleganbatikmadurabali.streamlit.app/" target="_blank" style="text-decoration: none; font-size: 14px; color: #333333; font-weight: 400; background-color: #EAEAEA; padding: 5px 12px; border-radius: 8px; box-shadow: 0px 1px 6px rgba(0, 0, 0, 0.05); display: flex; align-items: center; transition: all 0.3s ease;">
            <img src="https://img.icons8.com/ios-filled/50/333333/link.png" width="15" style="margin-right: 6px;"/> Style Batik Bali Merah
        </a>
        <a href="https://stylehijaubatikbali.streamlit.app/" target="_blank" style="text-decoration: none; font-size: 14px; color: #333333; font-weight: 400; background-color: #EAEAEA; padding: 5px 12px; border-radius: 8px; box-shadow: 0px 1px 6px rgba(0, 0, 0, 0.05); display: flex; align-items: center; transition: all 0.3s ease;">
            <img src="https://img.icons8.com/ios-filled/50/333333/link.png" width="15" style="margin-right: 6px;"/> Style Batik Bali Hijau
        </a>
    </div>
    <style>
        a:hover {
            transform: scale(1.03);
            box-shadow: 0px 3px 8px rgba(0, 0, 0, 0.1);
            background-color: #D1D1D1;
        }
    </style>
    """,
    unsafe_allow_html=True
)
