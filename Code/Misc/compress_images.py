from PIL import Image
import os

def compress_images(input_dir, output_dir, max_width=1200, quality=85):
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.png'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename.replace('.png', '.jpg'))
            
            try:
                img = Image.open(input_path)
                
                # Convert RGBA to RGB if necessary
                if img.mode == 'RGBA':
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[3])
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize if too large
                if img.width > max_width:
                    ratio = max_width / img.width
                    new_height = int(img.height * ratio)
                    img = img.resize((max_width, new_height), Image.LANCZOS)
                
                # Save as JPEG with compression
                img.save(output_path, 'JPEG', quality=quality, optimize=True)
                
                original_size = os.path.getsize(input_path) / 1024
                new_size = os.path.getsize(output_path) / 1024
                print(f"{filename}: {original_size:.0f}KB -> {new_size:.0f}KB ({(1-new_size/original_size)*100:.0f}% reduction)")
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Compress analysis plots
print("Compressing analysis_plots...")
compress_images('analysis_plots', 'analysis_plots_compressed')

print("\nCompressing industry_analysis_plots...")
compress_images('industry_analysis_plots', 'industry_analysis_plots_compressed')

print("\nDone! Upload the compressed folders to Overleaf.")
print("Remember to update image paths in LaTeX from .png to .jpg")
