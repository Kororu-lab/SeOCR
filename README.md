# SeOCR (Historical Korean Document OCR)

[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.50.0%2B-yellow)](https://huggingface.co/docs/transformers/index)

An OCR (Optical Character Recognition) system designed for digitizing historical Korean documents. This project is based on the ddobokki/ko-trocr model and optimized for the characteristics of historical Korean texts.

## Project Structure

```
SeOCR/
├── data/               # Data directory (for AI Hub dataset)
│   └── README.md       # Data structure and download guide
├── docs/               # Additional documentation
├── src/               # Source code
│   ├── __init__.py    # Package initialization
│   ├── config.py      # Configuration class
│   ├── dataset.py     # Dataset class
│   ├── model.py       # Model class
│   └── utils/
│       └── verify_data.py  # Data verification script
├── train.py           # Training script
├── requirements.txt   # Dependencies
└── README.md         # Project description
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/SeOCR.git
cd SeOCR
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation

1. Download 'Korean Historical Document OCR Dataset' from AI Hub
2. Verify data structure:
```bash
python -m src.utils.verify_data
```

For detailed data preparation instructions, refer to [data/README.md](data/README.md).

## Usage

### Training

```bash
python train.py
```

Trained models will be saved in the `seocr_checkpoints/` directory.

### Inference

```python
from src import SeOCR

# Initialize model
model = SeOCR()

# Load saved model
model.load_model('seocr_checkpoints/best_model')

# Extract text from image
text = model.predict('path/to/image.png')
print(text)
```

## Features

- [x] Korean TrOCR for historical documents
- [x] Support for manuscripts, woodblock prints, and movable type prints
- [x] Automatic train/validation split
- [x] Model checkpoint saving
- [x] Data verification tools
- [ ] Performance metrics (CER/WER)
- [ ] Data augmentation

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Citation

If you use this project in your research, please cite it as:

```bibtex
@software{SeOCR2024,
  author = {SeOCR Team},
  title = {SeOCR: Historical Korean Document OCR System},
  year = {2024},
  url = {https://github.com/Kororu-lab/SeOCR}
}
```

# SeOCR - Korean OCR System

This project implements a Korean OCR system using a two-stage approach: text detection followed by text recognition.

## Project Structure

```
SeOCR/
├── src/
│   └── ocr_model/
│       ├── config/
│       │   └── config.py
│       ├── detector/
│       │   ├── model.py
│       │   └── train.py
│       └── utils/
│           └── data_utils.py
├── predata/
│   ├── 필사본/
│   ├── 활자본/
│   └── 목판본/
├── logs/
│   └── detector/
└── checkpoints/
```

## Environment Setup

1. Create and activate conda environment:
```bash
conda create -n dl python=3.10
conda activate dl
```

2. Install required packages:
```bash
pip install torch torchvision
pip install tensorboard
pip install opencv-python
pip install pillow
pip install tqdm
pip install transformers
```

## Model Architecture

### Text Detector (CRAFT)
- Input: 256x256 RGB image
- Output: Text region scores, affinity scores, and detection scores
- Training:
  - Batch size: 1 (test mode)
  - Loss functions:
    - Region Loss: BCEWithLogitsLoss
    - Affinity Loss: BCEWithLogitsLoss
    - Detection Loss: BCEWithLogitsLoss

### Text Classifier (ViT-Large + CRNN)
- Input: 224x224 RGB image patches
- Output: Character class predictions
- Architecture:
  - ViT-Large: Global context understanding
  - CRNN: Sequence-based character recognition
  - LSTM: Sequence information processing

## Training

1. Train detector:
```bash
PYTHONPATH=src python src/ocr_model/detector/train.py
```

2. Train in test mode:
```bash
PYTHONPATH=src python src/ocr_model/detector/train.py --test
```

3. Monitor training:
```bash
tensorboard --logdir=logs/detector
```

## Inference

1. Run detection:
```bash
PYTHONPATH=src python src/ocr_model/detector/detect.py --source <image_path>
```

2. Run classification:
```bash
PYTHONPATH=src python src/ocr_model/classifier/classify.py --source <image_path>
```

## Model Checkpoints

- Detector: `checkpoints/detector_best.pth`
- Classifier: `checkpoints/classifier_best.pth`

## Training Results

- Model checkpoints are saved in `checkpoints/`
- Training logs are saved in `logs/detector/`
- Monitor the following metrics in Tensorboard:
  - Train/Val Loss
  - Train/Val Region Loss
  - Train/Val Affinity Loss
  - Train/Val Detection Loss
  - Learning Rate

## Notes

- Image size and batch size are adjusted to prevent CUDA out-of-memory errors
- Model size and batch size can be adjusted in `config.py` if memory issues occur
