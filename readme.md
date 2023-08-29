## This repository is built upon the paper titled "FS3: Few-Shot and Self-Supervised Framework for Efficient Intrusion Detection in Internet of Things Networks".

To execute this code, we have already provided the "requirements.txt" file which contains all the necessary packages for installation.

## Regarding the dataset: 
The WUSTL-EHMS dataset originates from a real-time Enhanced Healthcare Monitoring System (EHMS) that records network flow metrics and patients' biometrics. It comprises four main components: medical sensors, gateways, networks, and control with visualization. Patient data collected by sensors is transmitted through gateways, switches, and routers to the server. However, there's a potential risk of data interception, especially through man-in-the-middle attacks involving spoofing and data injection. The dataset includes 43 features, encompassing 35 network flow features and 8 patients' biometric features. Samples are labeled as "Normal" or "Attack", with attacker MAC addresses assigned a label of 1 and the rest as 0, based on the Source MAC address feature. Similar to the WUSTL-IIoT dataset, the WUSTL-EHMS dataset lacks separate training and testing datasets. To address this, the dataset is randomly divided into training and testing datasets since it lacks a timestamp feature. The dataset is located in the "dataset" folder.
## Pretrained model: 
A pre-trained model has been included in the "pretrain" folder.
## Few-shot learning and contrastive learning: 
To train the model contrastively using few-shot learning, run "contrastive_learning.py". The user has the flexibility to adjust the number of samples for model training; the default is 5. This implies selecting five samples from each class type to train the model using a contrastive loss function, specifically the Triplet loss function. Setting the margin to 0.1 allows the multi-similarity miner (MSN) to effectively distinguish hard positives and hard negatives within the similarity for an anchor.
## Classification using FAISS:
1. Running embeddingVectors.py to create embedding vectors of all the entries (training dataset and testing dataset)
2. Finally FAISS.py will provide the classification result.
