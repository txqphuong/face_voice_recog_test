import requests
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForCausalLM 
import torch
import warnings
import time
import os

warnings.filterwarnings("ignore")

class FlorenceVLM:
    def __init__(self, model_path, device=None):
        """
        Initialize Florence VLM model
        
        Args:
            model_path (str): Path to the Florence model
            device (str, optional): Device to run the model on. If None, will use CUDA if available
        """
        self.device = device if device else ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # Load model and processor
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True
        ).to(self.device)
        
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
    
    def process_image(self, image_path):
        """
        Process an image from local path or URL
        
        Args:
            image_path (str): Path to local image or URL
            
        Returns:
            PIL.Image: Processed image
        """
        if image_path.startswith(('http://', 'https://')):
            return Image.open(requests.get(image_path, stream=True).raw)
        else:
            return Image.open(image_path)
    
    def draw_and_save_boxes(self, image, boxes, labels, output_path):
        """
        Draw bounding boxes on the image and save it
        
        Args:
            image (PIL.Image): Original image
            boxes (list): List of bounding boxes [x1, y1, x2, y2]
            labels (list): List of labels for each box
            output_path (str): Path to save the output image
        """
        try:
            # Create a copy of the image to draw on
            draw_image = image.copy()
            draw = ImageDraw.Draw(draw_image)
            
            print(f"Drawing {len(boxes)} boxes with labels: {labels}")
            
            # Draw each box and label
            for box, label in zip(boxes, labels):
                x1, y1, x2, y2 = box
                print(f"Drawing box at coordinates: ({x1}, {y1}, {x2}, {y2}) with label: {label}")
                # Draw rectangle
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                # Draw label
                draw.text((x1, y1-10), label, fill="red")
            
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            if not os.path.exists(output_dir):
                print(f"Creating output directory: {output_dir}")
                os.makedirs(output_dir, exist_ok=True)
            
            # Save the image
            print(f"Attempting to save image to: {output_path}")
            draw_image.save(output_path)
            print(f"Successfully saved output image to: {output_path}")
            
            # Verify the file exists
            if os.path.exists(output_path):
                print(f"Verified file exists at: {output_path}")
                print(f"File size: {os.path.getsize(output_path)} bytes")
            else:
                print(f"Warning: File was not found at {output_path} after saving")
                
        except Exception as e:
            print(f"Error saving image: {str(e)}")
            raise

    def generate_response(self, image, prompt="<OD>", max_new_tokens=1024, num_beams=1, save_output=False, output_path=None):
        """
        Generate response from the model
        
        Args:
            image (PIL.Image): Input image
            prompt (str): Text prompt for the model
            max_new_tokens (int): Maximum number of tokens to generate
            num_beams (int): Number of beams for beam search
            save_output (bool): Whether to save the output image with boxes
            output_path (str): Path to save the output image if save_output is True
            
        Returns:
            dict: Parsed answer from the model
        """
        start_time = time.time()
        
        # Process inputs
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(self.device, self.torch_dtype)
        
        # Generate response
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=num_beams,
        )
        
        # Decode and parse response
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        print(f"Raw generated text: {generated_text}")
        
        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task=prompt,
            image_size=(image.width, image.height)
        )
        print(f"Full parsed answer: {parsed_answer}")
        
        # If OD task and save_output is True, draw and save boxes
        if prompt == "<OD>" and save_output and output_path:
            print(f"Parsed answer contains: {parsed_answer.keys()}")
            if "<OD>" in parsed_answer and "bboxes" in parsed_answer["<OD>"] and "labels" in parsed_answer["<OD>"]:
                print(f"Found {len(parsed_answer['<OD>']['bboxes'])} bboxes and {len(parsed_answer['<OD>']['labels'])} labels")
                self.draw_and_save_boxes(image, parsed_answer["<OD>"]["bboxes"], parsed_answer["<OD>"]["labels"], output_path)
            else:
                print("Warning: No boxes or labels found in parsed_answer")
                print(f"Full parsed_answer: {parsed_answer}")
        
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Generation time: {execution_time:.2f} seconds")
        
        return parsed_answer

# Example usage
if __name__ == "__main__":
    model_path = "Florence-2-base"
    florence = FlorenceVLM(model_path)
    
    # Test with local image
    image_path = "kitsuneOwner.png"
    print(f"Loading image from: {image_path}")
    if not os.path.exists(image_path):
        print(f"Error: Input image not found at {image_path}")
    else:
        print(f"Input image exists, size: {os.path.getsize(image_path)} bytes")
    
    image = florence.process_image(image_path)
    print(f"Image loaded successfully, size: {image.size}")
    
    # Generate response and save output image
    output_path = "kitsuneOwnerOutput.jpg"
    print(f"Will save output to: {output_path}")
    
    # Try different prompts for object detection
    prompts = [
        #"<OD>",
        "What does the image describe?",
    ]
    
    for prompt in prompts:
        print(f"\nTrying prompt: {prompt}")
        result = florence.generate_response(
            image,
            prompt=prompt,
            save_output=True,
            output_path=output_path
        )
        print(f"Result with prompt '{prompt}':", result)
