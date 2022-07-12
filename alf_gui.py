import streamlit as st
import pandas as pd
import numpy as np
import uuid
import logging
import sys
import os
import glob
import pathlib
import multiprocessing
import json
import time

# import scikit
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# import ALF modules
import alf.anotator
import alf.context_manager
import alf.d_manager
import alf.engine
import alf.evaluator
import alf.input_manager
import alf.ml_model
import alf.postprocess
import alf.preprocess
import alf.query_strategy

ContextProvider = alf.context_manager.ContextProvider
DbProvider = alf.d_manager.DbProvider

logging.basicConfig(
    stream=sys.stdout,
    format='[%(asctime)s]: %(message)s',
    level=logging.DEBUG
)

DEFAULT_FEATURES = [
    'BYTES',
    'BYTES_REV',
    'PACKETS',
    'PACKETS_REV',
    'SENT_PERCENTAGE',
    'RECV_PERCENTAGE',
    'IS_REQUEST_RESPONSE',
    'AVG_SECS_BETWEEN_PKTS',
    'OVERALL_DURATION_IN_SECS',
    'AVG_PKT_LEN',
    'PSH_RATIO'
]


MODELS = {
    'Random Forest': RandomForestClassifier(),
    'Decision Tree': DecisionTreeClassifier(max_depth=7),
    'K Neighbours': KNeighborsClassifier(n_neighbors=5)
}

def log_file_last_updated(filepath):
    p = pathlib.Path(filepath)
    if p.exists():
        return p.stat().st_mtime
    else:
        return 0

def get_file_last_line(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        if len(lines) > 0:
            return lines[-1]
        else:
            return ''

def check_pid(pid):        
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True



anotator = alf.anotator.AnotatorMiners()

st.sidebar.title('Select parameters')

model_selected = st.sidebar.selectbox('Model selection', list(MODELS.keys()))

nmax = st.sidebar.number_input('Max number of queried flows', min_value=0, value=1)

strategy_selected = st.sidebar.selectbox('Query strategy selection', ['Random', "Uncertainty"])

if strategy_selected == 'Uncertainty':
    threshold_selected = st.sidebar.number_input('Uncertainty threshold', max_value=1., min_value=0., value=0.6)

d0_file = st.sidebar.file_uploader("Choose a initial trainset file", accept_multiple_files=False, type="csv")
stream_files = st.sidebar.text_input('Folder with capture CSVs', placeholder='If empty, ./capture will be used')

features = st.sidebar.text_area("List of features to use, comma separated", value=",".join(DEFAULT_FEATURES))

run_id = st.sidebar.text_input('Run ID', placeholder="If empty, random ID will be used")
workdir = st.sidebar.text_input('Workdir', placeholder='If empty, /tmp/alf will be used')

if st.sidebar.button("Run", help="Run the ALF with selected params.", on_click=None, disabled=False):
    if not run_id:
        run_id = str(uuid.uuid4())
    if not workdir:
        workdir = '/tmp/alf'
    if not d0_file:
        st.write('No init train database file selected.')
        st.stop()
    if not strategy_selected:
        st.write('No query strategy selected.')
        st.stop()
    if not model_selected:
        st.write('No model selected.')
        st.stop()
    if not stream_files:
        stream_files = './capture'
    
    ContextProvider.create_context("file")
    ContextProvider.get_context().set_features(features.split(','))
    ContextProvider.get_context().set_experiment_id(run_id)
    ContextProvider.get_context().set_working_dir(workdir)
    d0 = pd.read_csv(d0_file)

    DbProvider.create_context(
        context_type="dataframe",
        d_0_path=d0)

    model = alf.ml_model.SupervisedMLModel(MODELS[model_selected])

    if strategy_selected == "Random":
        strategy = alf.query_strategy.RandomQueryStrategy(
            max_samples=nmax,
            anotator_obj=anotator,
            dry_run=True)
    elif strategy_selected == "Uncertainty":
        strategy = alf.query_strategy.UncertanityUnrankedBatch(
            max_samples=nmax,
            anotator_obj=anotator,
            dry_run=True,
            score_threshold=threshold_selected)
    
    postprocessor = alf.postprocess.PostprocessorIdentity()

    input_manager_obj = alf.input_manager.CSVFolderInputManager(stream_files)

    engine = alf.engine.Engine(
        preprocessor=alf.preprocess.PreprocessorIdentity(),
        postprocessor=postprocessor,
        ml_model_obj=model,
        query_strategy_obj=strategy,
        evaluator_obj=alf.evaluator.EvaluatorTestAnotatedAndAllPredicted(),
        input_manager_obj=input_manager_obj
    )

    process = multiprocessing.Process(target=engine.run)
    process.start()    
    st.sidebar.info(f'ALF was started with ID **{run_id}** in **{workdir}** as a process with PID **{process.pid}**')


    log_file_timestamp = 0
    scores = []
    placeholder = st.empty()
    while check_pid(process.pid):
        tmp = log_file_last_updated(f'{workdir}/metrics_{run_id}.json')
        if log_file_timestamp != tmp:
            try:
                log_file_timestamp = tmp
                last_record = get_file_last_line(f'{workdir}/metrics_{run_id}.json')
                f1_score = json.loads(last_record)['test_all_predicted']['f1']
                scores.append(f1_score)
                placeholder.line_chart(scores)
                process.join(0.1)
            except Exception as e:
                continue
    st.sidebar.success(f'ALF with ID **{run_id}** finished.')

    

