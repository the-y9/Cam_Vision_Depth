import cv2
import torch
import matplotlib.pyplot as plt


def initialize_midas(model_type='MiDaS_small'):
    """
    # Load the MiDaS model from the torch hub.
    """
    midas = torch.hub.load('intel-isl/MiDaS', model_type)
    midas.to('cpu')
    midas.eval()
    return midas


def initialize_transform():
    """
    Load the transformation pipeline for the MiDaS model.
    """
    transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
    return transforms.small_transform


def display_prediction(frame, output, window_name='CV2Frame'):
    """
    Display the original frame and the prediction using OpenCV and Matplotlib.
    """
    cv2.imshow(window_name, frame)
    plt.imshow(output, cmap='viridis')  # 'viridis' provides a good color map for depth
    plt.pause(0.00001)


def main():
    # Initialize MiDaS model and transform
    midas = initialize_midas()
    transform = initialize_transform()

    # Open video capture (webcam)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Failed to open webcam.")

    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to RGB
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Apply transformation
            imgbatch = transform(img).to('cpu')

            # Make prediction
            with torch.no_grad():
                prediction = midas(imgbatch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),  # Add a channel
                    size=img.shape[:2],  # Match input size
                    mode='bicubic',
                    align_corners=False
                ).squeeze()

                output = prediction.cpu().numpy()

            # Display results
            display_prediction(frame, output)

            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f'Error: {e}')            
    # finally:
        # Clean up resources
    cap.release()
    cv2.destroyAllWindows()
    plt.close()  # Ensure Matplotlib resources are cleaned up


if __name__ == "__main__":
    main()
