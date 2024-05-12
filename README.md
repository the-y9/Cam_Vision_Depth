# Vision Depth Project

This project demonstrates a simple computer vision application that captures video from a webcam, processes the frames, and uses a pre-trained PyTorch model (MiDaS) to perform depth estimation. The processed frames and predictions are displayed using OpenCV and Matplotlib.

## Requirements

- Python 3.6+
- OpenCV
- PyTorch
- Matplotlib

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/the-y9/Vision_Depth.git
   cd Vision_Depth

   ```

2. **Set Up a Virtual Environment** (optional but recommended)
   ```bash
   python -m venv .vd
   source .vd/bin/activate  # Linux/macOS
   .vd\Scripts\activate.bat  # Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   If there's no `requirements.txt`, manually install the necessary packages:
   ```bash
   pip install opencv-python torch matplotlib
   ```

## Usage

1. **Run the Script**
   ```bash
   python app.py
   ```

2. **Interacting with the Output**
   - The OpenCV window displays the webcam output.
   - The Matplotlib window displays the depth prediction.
   - To exit, press the 'q' key in the OpenCV window.

## Troubleshooting

- If the OpenCV window doesn't open, ensure the webcam is properly connected and working.
- If there are errors with package imports, double-check your virtual environment and reinstall the required packages.
- If the Matplotlib window keeps popping up, ensure proper cleanup with `plt.close()` after exiting.

## Acknowledgments

- This project uses the [MiDaS](https://github.com/intel-isl/MiDaS) model for depth estimation.
- Thanks to [OpenCV](https://opencv.org/) and [PyTorch](https://pytorch.org/) for providing the underlying libraries.
