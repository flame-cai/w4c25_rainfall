# Weather4Cast 2025 — Precipitation Nowcasting  

This repository contains the implementation of our submission for the **NeurIPS 2025 Weather4Cast Challenge**, focusing on **short-term precipitation nowcasting** using SEVIRI satellite data.  
Our pipeline leverages **ConvGRU** and **ConvLSTM** architectures to model spatiotemporal dependencies in brightness temperature imagery for future rainfall prediction.

---

## Setup Instructions  

### 1. Create a Python Environment  

It is recommended to use **Python 3.10+** and create a dedicated environment:

```bash
# Create a new virtual environment
python -m venv w4c_env

# Activate the environment
# On Windows:
w4c_env\Scripts\activate
# On macOS/Linux:
source w4c_env/bin/activate
