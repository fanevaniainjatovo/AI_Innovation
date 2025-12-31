# AI_Innovation
# Alcohol Consumption Prediction with MindSpore

## 📌 Project Overview

This project implements a **binary classification pipeline** using **MindSpore** to predict **alcohol consumption risk** from tabular data.  
The solution is designed as a **generic and reusable pipeline** that can later be extended to other substances (e.g., tobacco, cannabis, drugs) by following the **exact same steps**.

The current implementation focuses **only on alcohol consumption** (`abodalc` target).

---

## 🎯 Objectives

- Build a robust **binary classifier** for alcohol consumption
- Handle **highly imbalanced data**
- Ensure **clean dataset splitting** (train / validation / test)
- Provide **reliable evaluation metrics** beyond accuracy
- Enable **reproducible inference** using saved splits and checkpoints

---

## 🧠 Model Architecture

The model is a **Multi-Layer Perceptron (MLP)** implemented in MindSpore:

