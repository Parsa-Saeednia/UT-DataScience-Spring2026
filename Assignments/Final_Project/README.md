# Maestro Symbolic Music Generative Pipeline

## 🚀 Project Overview
This repository contains an advanced symbolic music processing and generative pipeline. The project utilizes the **MAESTRO** dataset to transition from raw symbolic music representation (MIDI) to structured musical intelligence. 

The architecture is built on a **Two-Stage Generative Pipeline**:
1. **Stage 1 (Clustering):** Unsupervised style clustering to identify stylistic markers (tonal distribution, transition rules) in musical compositions.
2. **Stage 2 (Generation):** A sequence-based LSTM model designed to generate structurally sound, coherent musical continuations conditioned on the identified style clusters.

---

## 🏗️ Architecture

### 1. Feature Engineering
We bypass traditional tabular prediction in favor of extracting "Musical DNA" from MIDI files. This pipeline computes:
* **Tonal Fingerprints:** Pitch class distributions and Markov transition matrices to quantify composer-specific grammar.
* **Expression Features:** Dynamic range (velocity variance), articulation (avg note duration), and rhythmic density (notes per second).
* **Sequential Tokens:** Chronological sequences formatted as [Pitch, Duration, Step, Velocity, Energy] to feed the LSTM.

### 2. Pipeline Flow
1. **Preprocessing:** Outlier removal (60s < duration < 1000s) and statistical scaling.
2. **Feature Extraction:** Atomic, modularized parsing of MIDI tracks.
3. **Persistence:** Data is serialized into .pkl artifacts (NumPy/Pandas format), optimized for direct ingestion by machine learning models.

## 📂 Project Structure

    ├── data/
    │   ├── dataframes/           # Serialized .pkl artifacts (final_features.pkl)
    │   └── maestro-v3.0.0/       # Raw MIDI dataset (Extract downloaded data here)
    ├── database/
    │   └── create_engineered_features.sql # Schema definition for portability
    ├── report/
    │   ├── database/             # Database architecture reports
    │   ├── docker/               # Docker configuration documentation
    │   ├── github_actions/       # CI/CD pipeline details
    │   └── kubernetes/           # K8s deployment reports
    ├── scripts/
    │   ├── database_connection.py # SQLAlchemy engine management
    │   └── feature_engineering.py # Pipeline logic (extraction, scaling, persistence)
    ├── Dockerfile                # Environment configuration for the image
    ├── EDA.ipynb                 # Exploratory Data Analysis & Proof of Concept
    ├── pod.yaml                  # Kubernetes Pod deployment configuration
    └── README.md

## ⚙️ Initial Setup

### 1. Data Preparation
You must manually download the MAESTRO dataset before running the pipeline.
1. Download the MAESTRO v3.0.0 dataset (MIDI format).
2. Extract the contents and place them inside the `data/maestro-v3.0.0/` directory. The structure should look like this: `data/maestro-v3.0.0/2004/`, `data/maestro-v3.0.0/2006/`, etc.

### 2. Environment Variables (.env)
To securely connect to the database without hardcoding credentials, create a `.env` file in the root directory of the project and define the following variables:

    DB_HOST=host.docker.internal
    DB_PORT=3306
    DB_USER=root
    DB_PASSWORD=your_secure_password
    DB_NAME=maestro_db

*(Note: Use `host.docker.internal` if your MySQL server is running on the host machine and you are using Docker Desktop).*

## 🚀 How to Run the Project (Local Kubernetes Deployment)

Follow these step-by-step instructions to build the Docker image and deploy the data pipeline inside the local Kubernetes cluster provided by Docker Desktop.

### Prerequisites
1. Ensure Docker Desktop is running.
2. Ensure Kubernetes is enabled in Docker Desktop Settings (Settings > Kubernetes > Enable Kubernetes).

---

### Step 1: Clean Network Variables & Target the Correct Context
Open your WSL/Linux terminal and clear any proxy variables that might interfere with local Kubernetes communication, then switch to the Docker Desktop context:

    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
    kubectl config use-context docker-desktop

### Step 2: Verify Kubernetes Cluster Status
Navigate to your project directory and check if the Kubernetes node is ready:

    cd /mnt/d/University/DS/CAs/UT-DataScience-Spring2026/Assignments/Final_Project
    kubectl get nodes

*(You should see docker-desktop listed with a Ready status).*

### Step 3: Build the Docker Image
Build the container image using the provided Dockerfile. We tag it as maestro-pipeline:latest so Kubernetes can easily locate it:

    docker build -t maestro-pipeline:latest .

### Step 4: Deploy the Pod
Apply the Kubernetes configuration file to launch the pipeline pod:

    kubectl apply -f pod.yaml

### Step 5: Monitor Execution
Check the status of the pod to ensure it is running or completed:

    kubectl get pods

To view the output and verify that the feature engineering pipeline executed successfully, stream the logs:

    kubectl logs music-pipeline-pod


## 📊 Analytics & Justification

The EDA phase (EDA.ipynb) acts as the mathematical validation for our model architecture:
* **Preprocessing Proof:** Histograms demonstrate the necessity of the 60s minimum duration cutoff to ensure sufficient sequence data for the LSTM.
* **Temporal Bias Detection:** Heatmaps confirm the need to drop 'year' to prevent temporal data leakage.
* **Style Separability:** Markov chain transitions for different composers visually prove that the stylistic features we extract create distinct, separable clusters for our Stage-1 model.

***
*Developed for the University of Tehran, Spring 2026 - Data Science CA4*