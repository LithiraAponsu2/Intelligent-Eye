import os
from moviepy.editor import VideoFileClip

def convert_directory_mov_to_mp4(input_dir=".", output_dir="."):
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # List all MOV files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".MOV"):
            # Construct full file paths
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.mp4")
            
            # Convert the file
            clip = VideoFileClip(input_path)
            clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
            print(f"Converted {filename} to MP4.")
            

# Just call the function without parameters for current directory operations
convert_directory_mov_to_mp4()
