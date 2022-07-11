import streamlit as st
import pandas as pd
import numpy as np
import uuid


st.sidebar.title('Select parameters')

model = st.sidebar.selectbox('Model selection', ['Random Forest', 'Decision Tree'])
strategy = st.sidebar.selectbox('Query strategy selection', ['Random', "Uncertainty"])

if strategy == 'Uncertainty':
    threshold = st.sidebar.number_input('Uncertainty threshold', max_value=1., min_value=0., value=0.6)

d0_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
run_id = st.sidebar.text_input('Run ID', placeholder="If empty, random ID will be used")
workdir = st.sidebar.text_input('Workdir', placeholder='If empty, /tmp/ will be used')

if st.sidebar.button("Run", help="Run the ALF with selected params.", on_click=None, disabled=False):
    if not run_id:
        run_id = str(uuid.uuid4())
    if not workdir:
        workdir = '/tmp/'
    if not d0_file:
        st.write('No init train database file selected.')
        st.stop()
    st.sidebar.success(f'Running ALF with ID **{run_id}** in **{workdir}**')
    d0 = pd.read_csv(d0_file)
    st.write(d0)

