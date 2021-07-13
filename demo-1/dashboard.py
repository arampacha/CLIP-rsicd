import dashboard_text2image
import dashboard_image2image

import streamlit as st

PAGES = {
    "Text to Image": dashboard_text2image,
    "Image to Image": dashboard_image2image
}
st.sidebar.title("Navigation")

selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()
