# ALF - Active Learning Framework

Recent network traffic classification methods benefit from machine learning (ML) technology. However, there are many challenges due to use of ML, such as: lack of high-quality annotated datasets, data-drifts and other effects causing aging of datasets and ML models, high volumes of network traffic etc. We presents a novel Active Learning Framework (ALF) to address this topic. ALF provides prepared software components that can be used to deploy an active learning loop and maintain an ALF instance that continuously evolves a dataset and ML model automatically. The resulting solution is deployable for IP flow-based analysis of high-speed (100 Gb/s) networks, and also supports research experiments on different strategies and methods for annotation, evaluation, dataset optimization, etc.

<p align="center">
<img src="docs/alf.png" width="300" height="300">
</p>

## Architecture

ALF is a framework that provides a set of software components that can be used to deploy an active learning loop and maintain an ALF instance that continuously evolves a dataset and ML model automatically. The resulting solution is deployable for IP flow-based analysis of high-speed (100 Gb/s) networks, and also supports research experiments on different strategies and methods for annotation, evaluation, dataset optimization, etc.

Previous version of ALF was presented in [1]. Previous version was implemented in `Python 3.10` as single application. The new version is characterized by decomposition to smaller independent modules implemented in `Python` and `C++` for heavy performance parts -- including AL Loop.

Main parts of ALF are:
* `Feature generator` (C++), NEMEA-dependent module
* `Flow Sampler` (C++), NEMEA-dependent module
* `AL Core` (C++), independent module
* `Annotator` (Python), independent module
* `QoD` (Python), independent module

`Feature generator` and `Flow sampler` are NEMEA-dependent modules. They are used to generate features and samples from network traffic. `AL Core` is independent module that is used to run AL loop. `Annotator` is independent module that is used to annotate samples. `QoD` is independent module that is used to evaluate quality of dataset. 

Sharing of data between `AL Core`, `Annotator` and `QoD` is done via `SQLite` database. `AL Core` is responsible for Active Learning Loop, `Annotator` annotates selected flows and `QoD` evaluates quality of dataset.

`SQLite` is used as a compromise between file based and database based solution.

## SQLite database structure

* ID - unique ID of flow
* Features set - columns from interval `[1; cols-7]` are features of flow
* Class - column `cols-6` is label from {-1, 0, .. , n} where n is number of classes, `-1` means that flow is not labeled. Source of truth.
* Annotate - column `cols-5` is flag that indicates if flow should be annotated, `0` means that flow should not be annotated, `1` means that flow should be annotated
* Annotation_time - column `cols-4` is time when flow was annotated
* Predicted class - column `cols-3` is predicted class of flow
* Predicted probability - column `cols-2` is predicted probability of predicted class of flow
* Meta - column `cols-1` is meta information about flow, string, can be used for any purpose, in ALF not standardized

## Installation
```shell
cmake .
make
```

### Prerequisites
* `mlpack`
* `sqlite3`
* `armadillo`
* `ensmallen`