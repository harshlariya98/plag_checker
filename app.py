import streamlit as st

st.set_page_config(page_title="KollegeApply Plag Checker", layout="wide",
                   initial_sidebar_state="collapsed")
st.markdown("""
<style>
[data-testid="stSidebar"], [data-testid="stSidebarNav"],
header, #MainMenu { display: none !important; }
.stApp { background: #fff; }
</style>
""", unsafe_allow_html=True)

st.switch_page("pages/internal-plag-checker.py")
