# CASS-Net
## 🏗 Model Architecture

CASS-Net is designed for the performance-efficiency trade-off required in emergency settings:

*   **Encoder:** Pre-trained EfficientNet-Lite0 (Adapted for 4-channel input).
*   **Decoder:** Depthwise Separable Convolutions (DSConv) to reduce parameters.
*   **Attention:** Dual mechanism with Spatial Attention Gates (AG) and SE-Blocks.
*   **Input Strategy:** 2.5D Multi-slice (Target + Neighbors) with Stroke/Brain windowing.
    
## 📂 Repository Structure

```text
CASS-Net/
├── models/
│   ├── cass_net.py        # Main Model Architecture
│   └── layers.py          # Custom Blocks (SE, AG, DSConv)
├── utils/
│   ├── dataset.py         # 2.5D Data Loader & Windowing
│   └── losses.py          # Composite Loss (Focal Tversky + Dice)
├── train.py               # Training Script
├── requirements.txt       # Dependencies
└── README.md              # Documentation

🚀 Getting Started
1. Prerequisites
Install the required packages:
code
Bash
pip install -r requirements.txt
