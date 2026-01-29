# CASS-Net
## 🏗 Model Architecture

CASS-Net is designed for the performance-efficiency trade-off required in emergency settings:

*   **Encoder:** Pre-trained EfficientNet-Lite0 (Adapted for 4-channel input).
*   **Decoder:** Depthwise Separable Convolutions (DSConv) to reduce parameters.
*   **Attention:** Dual mechanism with Spatial Attention Gates (AG) and SE-Blocks.
*   **Input Strategy:** 2.5D Multi-slice (Target + Neighbors) with Stroke/Brain windowing.
    
## 📂 Repository Structure

CASS-Net/
├── models/
│   ├── cass_net.py
│   ├── layers.py
│   └── __init__.py
├── utils/
│   ├── dataset.py
│   └── losses.py
├── train.py
└── requirements.txt
