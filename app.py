import cv2
import torch

def initialize_midas(model_type='MiDaS_small'):
    """
    Load the MiDaS model from the torch hub.
    """
    midas = torch.hub.load('intel-isl/MiDaS', model_type)
    midas.eval()
    return midas


def initialize_transform():
    """
    Load the transformation pipeline for the MiDaS model.
    """
    transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
    return transforms.small_transform


def display_prediction(frame, depth_map):
    """
    Display original RGB frame and depth prediction side-by-side using OpenCV.
    """
    # Normalize the depth map to 0-255 and apply a colormap
    depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_colored = cv2.applyColorMap(depth_normalized.astype('uint8'), cv2.COLORMAP_MAGMA)

    # Show both the original frame and depth map
    cv2.imshow('RGB Frame', frame)
    cv2.imshow('Depth Map', depth_colored)


def main():
    # Set device once
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model and transform
    midas = initialize_midas().to(device)
    transform = initialize_transform()

    # Start webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Unable to access the webcam.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize for performance (optional)
            resized_frame = cv2.resize(frame, (320, 240))

            # Convert to RGB
            img_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

            # Prepare image batch
            input_batch = transform(img_rgb).to(device)

            # Inference
            with torch.no_grad():
                prediction = midas(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img_rgb.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

                depth_map = prediction.cpu().numpy()

            # Display depth
            display_prediction(resized_frame, depth_map)

            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error: {e}")

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()