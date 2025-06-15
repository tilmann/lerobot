import cv2
import rerun.sdk as rr
import time

def main():
    # Initialize Rerun - updated API
    rr.init("camera_feed")  # Use init instead of connect
    
    # Initialize camera (0 is usually the default camera)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to capture image")
                break
            
            # Log the frame to Rerun
            rr.log("camera", rr.Image(frame))
            
            # Small delay to prevent overwhelming the viewer
            time.sleep(0.03)
            
    except KeyboardInterrupt:
        print("Stopping camera feed...")
    
    finally:
        # Release the camera
        cap.release()
        print("Camera released.")

if __name__ == "__main__":
    main()