import cv2
import os

def extract_frames(video_path, output_folder, interval_sec=1):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open video file using the passed argument
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("Error: Could not get FPS of the video.")
        return

    interval_frames = int(fps * interval_sec)

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % interval_frames == 0:
            filename = os.path.join(output_folder, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Done! Extracted {saved_count} frames every {interval_sec} seconds.")

# ✅ Example usage with full path to the video file
video_path = r"E:\MyProjects\AASA IT SOLUTION\Knuckles Tree\FastRCNN1.5\AasaITGArden\IMG_2160.MOV"
output_folder = r"E:\MyProjects\AASA IT SOLUTION\Knuckles Tree\FastRCNN1.5\output_images"

extract_frames(video_path, output_folder, interval_sec=1)
