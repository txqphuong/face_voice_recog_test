import cv2
import time
from datetime import datetime
import os
from testFlorence import FlorenceVLM
import numpy as np
import threading
import torch
import argparse
import json
import fcntl
torch.classes.__path__ = []

class CameraDescriptor:
    def __init__(self, model_path, camera_id=0, output_file="descriptions.json", images_dir="tmp/cameraVlm"):
        """
        Initialize the camera descriptor
        
        Args:
            model_path (str): Path to the Florence model
            camera_id (int): Camera device ID (default: 0 for primary camera)
            output_file (str): JSON file to save descriptions
            images_dir (str): Directory to save captured images
        """
        self.camera_id = camera_id
        self.output_file = output_file
        self.images_dir = images_dir
        self.florence = FlorenceVLM(model_path)
        self.last_description = "Waiting for description..."
        self.processing = False
        self.lock = threading.Lock()
        self.exit_button_clicked = False
        
        # Create images directory if it doesn't exist
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Initialize or load existing descriptions
        self.descriptions = self.load_descriptions()
        
        # Set up mouse callback
        cv2.namedWindow('Live Camera Feed')
        cv2.setMouseCallback('Live Camera Feed', self.mouse_callback)
    
    def load_descriptions(self):
        """Load existing descriptions from JSON file if it exists"""
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, 'r') as f:
                    # Acquire a shared lock for reading
                    fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                    try:
                        data = json.load(f)
                    finally:
                        # Release the lock
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                    return data
            except json.JSONDecodeError:
                print(f"Warning: Could not parse {self.output_file}, starting with empty descriptions")
                return {"descriptions": []}
        return {"descriptions": []}
    
    def save_descriptions(self):
        """Save descriptions to JSON file, preserving existing data"""
        try:
            # Create a backup of the existing file if it exists
            if os.path.exists(self.output_file):
                backup_file = f"{self.output_file}.bak"
                with open(self.output_file, 'r') as src, open(backup_file, 'w') as dst:
                    # Acquire a shared lock for reading
                    fcntl.flock(src.fileno(), fcntl.LOCK_SH)
                    try:
                        dst.write(src.read())
                    finally:
                        # Release the lock
                        fcntl.flock(src.fileno(), fcntl.LOCK_UN)
            
            # Save the new data with an exclusive lock
            with open(self.output_file, 'w') as f:
                # Acquire an exclusive lock for writing
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    json.dump(self.descriptions, f, indent=2)
                finally:
                    # Release the lock
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                
            # Remove backup if save was successful
            if os.path.exists(f"{self.output_file}.bak"):
                os.remove(f"{self.output_file}.bak")
                
        except Exception as e:
            print(f"Error saving descriptions: {str(e)}")
            # Restore from backup if save failed
            if os.path.exists(f"{self.output_file}.bak"):
                os.replace(f"{self.output_file}.bak", self.output_file)
            raise

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if click is within exit button area (top-right corner)
            if x >= 610 and x <= 630 and y >= 10 and y <= 25:
                self.exit_button_clicked = True

    def create_description_overlay(self, frame, description):
        """
        Create an overlay with the description text and exit button
        
        Args:
            frame: The camera frame
            description: The text to display
            
        Returns:
            The frame with description overlay
        """
        try:
            if frame is None or frame.size == 0:
                print("Warning: Invalid frame received in create_description_overlay")
                return None

            # Create a copy of the frame
            overlay = frame.copy()
            
            # Define text parameters
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_thickness = 2
            text_color = (255, 255, 255)  # White text
            bg_color = (0, 0, 0)  # Black background
            
            # Split description into multiple lines if it's too long
            words = str(description).split()
            lines = []
            current_line = []
            
            for word in words:
                current_line.append(word)
                # Check if adding the next word would make the line too long
                if len(' '.join(current_line)) > 50:
                    lines.append(' '.join(current_line[:-1]))
                    current_line = [word]
            
            if current_line:
                lines.append(' '.join(current_line))
            
            # Add semi-transparent background
            overlay_height = 30 * (len(lines) + 1)  # Height for text + padding
            overlay[0:overlay_height, :] = cv2.addWeighted(
                overlay[0:overlay_height, :], 0.7,
                np.full((overlay_height, overlay.shape[1], 3), bg_color, dtype=np.uint8), 0.3,
                0
            )
            
            # Add text
            y_position = 30
            for line in lines:
                cv2.putText(overlay, line, (10, y_position), font, font_scale, text_color, font_thickness)
                y_position += 30
            
            # Add exit button (smaller size and moved right)
            button_color = (0, 0, 255)  # Red color
            cv2.rectangle(overlay, (610, 10), (630, 25), button_color, -1)
            cv2.putText(overlay, "X", (615, 22), font, 0.5, (255, 255, 255), 1)
            
            return overlay
        except Exception as e:
            print(f"Error in create_description_overlay: {str(e)}")
            return frame

    def process_frame(self, frame):
        """
        Process frame in a separate thread 
        """
        if self.processing or frame is None:
            return None

        self.processing = True
        try:
            # Save the captured frame
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_filename = f"image_{timestamp}.jpg"
            image_path = os.path.join(self.images_dir, image_filename)
            cv2.imwrite(image_path, frame)
            print(f"\nCaptured image saved to: {image_path}")
            
            # Convert frame to RGB (Florence expects RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image for Florence
            from PIL import Image
            pil_image = Image.fromarray(frame_rgb)
            
            prompt = "What does the image describe?"
            # Get description from Florence
            result = self.florence.generate_response(
                pil_image,
                prompt=prompt,
                save_output=False
            )
            
            description = result[prompt]
            with self.lock:
                self.last_description = description
                
                # Append new description to the list
                new_entry = {
                    "timestamp": timestamp,
                    "image_path": image_path,
                    "description": description
                }
                
                self.descriptions["descriptions"].append(new_entry)
                
                self.save_descriptions()
                
            print(f"Description: {description}")
            
            return description
            
        except Exception as e:
            print(f"Error in process_frame: {str(e)}")
            return None
        finally:
            self.processing = False
        
    def capture_and_describe(self):
        """
        Capture images from camera every 3 seconds and describe them using Florence VLM
        """
        # Initialize camera
        cap = cv2.VideoCapture(self.camera_id)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {self.camera_id}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        last_capture_time = 0
        capture_interval = 3  # seconds
        
        try:
            while True:
                # Capture frame
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("Error: Failed to capture frame")
                    time.sleep(0.1)  # Add small delay to prevent CPU overload
                    continue
                
                current_time = time.time()
                
                # Process frame every 3 seconds in a separate thread
                if current_time - last_capture_time >= capture_interval:
                    thread = threading.Thread(target=self.process_frame, args=(frame.copy(),))
                    thread.start()
                    last_capture_time = current_time
                
                try:
                    # Create overlay with description
                    with self.lock:
                        display_frame = self.create_description_overlay(frame, self.last_description)
                    
                    if display_frame is not None:
                        # Display the live camera feed
                        cv2.imshow('Live Camera Feed', display_frame)
                        
                        # Check for exit button click
                        if self.exit_button_clicked:
                            break
                        
                        # Break loop if 'q' is pressed (keeping as backup)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                except Exception as e:
                    print(f"Error displaying frame: {str(e)}")
                    time.sleep(0.1)  # Add small delay to prevent CPU overload
                
        except KeyboardInterrupt:
            print("\nStopping camera capture...")
        except Exception as e:
            print(f"Error in capture_and_describe: {str(e)}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Camera released")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Camera Descriptor using Florence VLM')
    parser.add_argument('--model_path', type=str, default="Florence-2-base-ft",
                      help='Path to the Florence model directory')
    parser.add_argument('--camera_id', type=int, default=0,
                      help='Camera device ID (default: 0 for primary camera)')
    parser.add_argument('--output_file', type=str, default="tmp/cameraVlm/descriptions.json",
                      help='JSON file to save descriptions (default: descriptions.json)')
    parser.add_argument('--images_dir', type=str, default="tmp/cameraVlm/captured_images",
                      help='Directory to save captured images (default: captured_images)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create output directory for JSON file if it doesn't exist
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize with provided arguments
    descriptor = CameraDescriptor(
        model_path=args.model_path,
        camera_id=args.camera_id,
        output_file=args.output_file,
        images_dir=args.images_dir
    )
    
    # Start capturing and describing
    print("Starting camera capture and description...")
    print("Press 'q' to quit")
    descriptor.capture_and_describe() 