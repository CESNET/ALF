import streamlit as st
import pandas as pd
import numpy as np
import uuid


DEFAULT_FEATURES = [
    'bytes_rev',
    'bytes',
    'packets',
    'packets_rev',
    'packets_sum',
    'bytes_ration',
    'num_pkts_ration',
    'time',
    'av_pkt_size',
    'av_pkt_size_rev',
    'var_pkt_size',
    'var_pkt_size_rev',
    'median_pkt_size',
    'median_pkt_size_rev',
    'mindelay',
    'avgdelay',
    'maxdelay',
    'bursts',
    'fizzles',
    'time_leap_ration',
    'autocorr',
    'stSum',
    'ndSum',
    'rdSum'
]


st.sidebar.title('Select parameters')

model = st.sidebar.selectbox('Model selection', ['Random Forest', 'Decision Tree'])

nmax = st.sidebar.number_input('Max number of queried flows', min_value=0, value=1)

strategy = st.sidebar.selectbox('Query strategy selection', ['Random', "Uncertainty"])

if strategy == 'Uncertainty':
    threshold = st.sidebar.number_input('Uncertainty threshold', max_value=1., min_value=0., value=0.6)

d0_file = st.sidebar.file_uploader("Choose a initial trainset file", accept_multiple_files=False, type="csv")
stream_files = st.sidebar.file_uploader("Choose a stream data", accept_multiple_files=True, type="csv")

features = st.sidebar.text_area("List of features to use, comma separated", value=",".join(DEFAULT_FEATURES))

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

