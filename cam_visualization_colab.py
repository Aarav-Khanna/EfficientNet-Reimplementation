import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from PIL import Image
import os
import cv2
import sys
from matplotlib.colors import LinearSegmentedColormap
from google.colab import drive

# Mount Google Drive if not already mounted
if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')

# Import from EfficientNet project
sys.path.append('/content/drive/MyDrive/EfficientNet-Reimplementation')
from models.efficientnet import efficientnet_b0
from parameter_search import create_scaled_model

# Create a colormap similar to the one in the paper
def get_cam_colormap():
    """Create a colormap similar to the one used in the EfficientNet paper's CAM visualization"""
    colors = []
    for i in range(256):
        if i < 85:  # blue to cyan
            r = 0
            g = i * 3
            b = 255
        elif i < 170:  # cyan to yellow
            r = (i - 85) * 3
            g = 255
            b = 255 - (i - 85) * 3
        else:  # yellow to red
            r = 255
            g = 255 - (i - 170) * 3
            b = 0
        colors.append((r / 255, g / 255, b / 255))
    return LinearSegmentedColormap.from_list('custom_colormap', colors)

class CAMExtractor:
    """Class to extract Class Activation Maps from EfficientNet models"""
    def __init__(self, model, target_layer='blocks.15', device='cuda'):
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Register forward hook to get feature maps
        self.feature_maps = None
        self.hook_handle = None
        
        def hook_fn(module, input, output):
            self.feature_maps = output.detach()
        
        # Find the target layer
        for name, module in self.model.named_modules():
            if name == target_layer:
                self.hook_handle = module.register_forward_hook(hook_fn)
                break
        
        if self.hook_handle is None:
            print(f"Target layer {target_layer} not found. Available layers:")
            for name, _ in self.model.named_modules():
                print(name)
            raise ValueError(f"Target layer {target_layer} not found.")
    
    def get_cam(self, input_tensor, class_idx=None):
        """
        Generate a CAM for a specific class
        
        Args:
            input_tensor: The input image tensor
            class_idx: Class index to generate CAM for. If None, uses the predicted class.
        
        Returns:
            cam: The class activation map
            pred_class: The predicted class
            confidence: The confidence score for the predicted class
        """
        # Forward pass
        with torch.no_grad():
            input_tensor = input_tensor.to(self.device)
            output = self.model(input_tensor)
            
            # Get prediction
            confidence, pred_class = torch.max(F.softmax(output, dim=1), dim=1)
            if class_idx is None:
                class_idx = pred_class.item()
            
            # Get weights from the final layer for the specific class
            # For EfficientNet, this is the classifier layer
            weights = self.model.classifier[3].weight[class_idx].cpu().data.numpy()
            
            # Generate CAM
            feature_maps = self.feature_maps.cpu().data.numpy().squeeze()
            
            # If feature maps have 4 dimensions [batch, channels, height, width]
            if len(feature_maps.shape) == 4:
                feature_maps = feature_maps.squeeze(0)  # Remove batch dimension
            
            # Create cam of the correct shape
            cam = np.zeros(feature_maps.shape[1:], dtype=np.float32)
            
            # Weight sum
            for i, w in enumerate(weights):
                # Make sure we don't go out of bounds
                if i < feature_maps.shape[0]:
                    cam += w * feature_maps[i]
            
            # ReLU and normalize
            cam = np.maximum(cam, 0)  # ReLU
            cam = cv2.resize(cam, (input_tensor.shape[2], input_tensor.shape[3]))
            cam = cam - np.min(cam)
            cam = cam / (np.max(cam) + 1e-10)  # Normalize
            
            return cam, pred_class.item(), confidence.item()
    
    def __del__(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()

def preprocess_image(image_path, input_size=224, device='cuda'):
    """Preprocess an image for the model"""
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    return input_tensor, image
    
def generate_cam_visualization(model, image_path, input_size, class_idx=None, 
                              target_layer='blocks.15', device='cuda'):
    """Generate a CAM visualization for an image"""
    # Preprocess the image
    input_tensor, original_image = preprocess_image(image_path, input_size, device)
    
    # Extract CAM
    cam_extractor = CAMExtractor(model, target_layer, device)
    cam, pred_class, confidence = cam_extractor.get_cam(input_tensor, class_idx)
    
    # Resize the original image
    original_image = original_image.resize((input_size, input_size))
    
    return cam, original_image, pred_class, confidence

def create_models_with_different_scaling():
    """Create models with different scaling approaches"""
    num_classes = 100  # CIFAR-100
    models = {
        'baseline': efficientnet_b0(num_classes=num_classes, input_size=224),
        'deeper': create_scaled_model(4.0, 1.0, 1.0, num_classes=num_classes, input_size=224),
        'wider': create_scaled_model(1.0, 2.0, 1.0, num_classes=num_classes, input_size=224),
        'higher_resolution': create_scaled_model(1.0, 1.0, 2.0, num_classes=num_classes, input_size=224),
        'compound_scaling': create_scaled_model(1.4, 1.2, 1.3, num_classes=num_classes, input_size=224)
    }
    return models

def visualize_cams(images_path, output_path, device='cuda'):
    """Generate CAM visualizations for multiple images and scaling approaches"""
    # Create models
    models = create_models_with_different_scaling()
    
    # Check if the images path exists
    if not os.path.exists(images_path):
        print(f"Error: Images path '{images_path}' does not exist.")
        print(f"Current working directory: {os.getcwd()}")
        return
    
    # Get list of images
    image_files = [f for f in os.listdir(images_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Check if we found any images
    if not image_files:
        print(f"Error: No images found in '{images_path}'.")
        print(f"Files in directory: {os.listdir(images_path)}")
        return
    
    print(f"Found {len(image_files)} images: {image_files}")
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Set up the plot
    fig_height = 4 * len(image_files)
    fig, axs = plt.subplots(len(image_files), 6, figsize=(18, fig_height))
    
    # Handle case with single image and multiple images differently
    if len(image_files) == 1:
        axs = [axs]  # Wrap in list for consistent indexing
    
    # Set column titles
    column_titles = ['original image', 'baseline model', 'deeper (d=4)', 
                    'wider (w=2)', 'higher resolution (r=2)', 'compound scaling']
    
    for col, title in enumerate(column_titles):
        axs[0][col].set_title(title, fontsize=12)
    
    # Process each image
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(images_path, image_file)
        print(f"Processing image: {image_path}")
        
        # Add row label (image name without extension)
        img_name = os.path.splitext(image_file)[0]
        axs[i][0].set_ylabel(img_name, rotation=0, labelpad=40, 
                            fontsize=12, va='center', ha='right')
        
        # Show original image
        original_img = Image.open(image_path).convert('RGB')
        axs[i][0].imshow(original_img)
        axs[i][0].axis('off')
        
        # Process with each model
        for j, (model_name, model) in enumerate(models.items(), start=1):
            print(f"  - Processing with {model_name} model")
            # Adjust input size based on resolution scaling
            if model_name == 'higher_resolution':
                input_size = 448  # 224 * 2
            elif model_name == 'compound_scaling':
                input_size = int(224 * 1.3)  # as per the paper
            else:
                input_size = 224
                
            # Generate CAM
            try:
                cam, _, pred_class, confidence = generate_cam_visualization(
                    model, image_path, input_size, class_idx=None, 
                    target_layer='blocks.15', device=device
                )
                
                # Display CAM
                axs[i][j].imshow(cam, cmap=get_cam_colormap())
                axs[i][j].axis('off')
                
                # Add class prediction as text
                if pred_class is not None:
                    axs[i][j].text(5, 15, f"Class: {pred_class}", 
                                color='white', fontsize=10, 
                                bbox=dict(facecolor='black', alpha=0.5))
                
                print(f"    Success! Class: {pred_class}, Confidence: {confidence:.2f}")
            except Exception as e:
                print(f"    Error processing {image_file} with {model_name}: {e}")
                axs[i][j].text(0.5, 0.5, "Error", 
                              ha='center', va='center', 
                              transform=axs[i][j].transAxes)
                axs[i][j].axis('off')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'cam_visualization.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_path, 'cam_visualization.pdf'), bbox_inches='tight')
    print(f"CAM visualization saved to {output_path}")
    plt.show()

def main():
    # Set default paths specifically for Google Colab
    default_images_path = '/content/drive/MyDrive/EfficientNet-Reimplementation/sample_images'
    default_output_path = '/content/drive/MyDrive/EfficientNet-Reimplementation/results/cam_visualization'
    
    parser = argparse.ArgumentParser(description='Generate CAM visualizations for different scaling approaches')
    parser.add_argument('--images_path', type=str, default=default_images_path,
                        help=f'Path to folder containing sample images (default: {default_images_path})')
    parser.add_argument('--output_path', type=str, default=default_output_path,
                        help=f'Path to save visualization results (default: {default_output_path})')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda or cpu)')
    args = parser.parse_args()
    
    print(f"Looking for images in: {args.images_path}")
    print(f"Output will be saved to: {args.output_path}")
    
    visualize_cams(args.images_path, args.output_path, args.device)

if __name__ == '__main__':
    main() 