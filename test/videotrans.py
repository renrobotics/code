import os
import argparse
from moviepy.editor import VideoFileClip
from pathlib import Path

def restore_normal_speed(input_folder, output_folder):
    """
    Processes all video files in an input folder to restore their normal speed
    (assuming they are currently 2x speed) and saves them to an output folder.

    Args:
        input_folder (str): The path to the folder containing the 2x speed videos.
        output_folder (str): The path to the folder where the normal speed videos will be saved.
    """
    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)
    print(f"Output directory '{output_folder}' is ready.")

    # List of common video file extensions to look for
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']

    # Get a list of video files from the input folder
    try:
        video_files = [f for f in os.listdir(input_folder) if Path(f).suffix.lower() in video_extensions]
    except FileNotFoundError:
        print(f"[ERROR] The input folder '{input_folder}' was not found.")
        return

    if not video_files:
        print(f"No video files found in '{input_folder}'. Please check the path and file extensions.")
        return

    print(f"Found {len(video_files)} video(s) to process.")

    for filename in video_files:
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        print(f"\nProcessing '{input_path}'...")

        try:
            # Load the video clip
            with VideoFileClip(input_path) as clip:
                # To restore a 2x speed video to normal (1x), we apply a speed factor of 0.5.
                # The final clip's duration will be clip.duration / 0.5 = clip.duration * 2.
                print(f"  Original duration: {clip.duration:.2f} seconds (at 2x speed)")
                normal_speed_clip = clip.speedx(0.5)
                print(f"  New duration: {normal_speed_clip.duration:.2f} seconds (at 1x speed)")

                # Write the result to the output file
                # The 'codec' parameter is important for compatibility. 'libx264' is a good default for .mp4.
                # The 'threads' parameter can speed up writing on multi-core CPUs.
                # The 'logger=None' suppresses verbose ffmpeg output. For debugging, set to 'bar'.
                print(f"  Saving to '{output_path}'...")
                normal_speed_clip.write_videofile(output_path, codec='libx264', threads=4, logger=None)
                print(f"  Successfully saved.")

        except Exception as e:
            print(f"  [ERROR] An error occurred while processing {filename}: {e}")

if __name__ == '__main__':
    # Set up argument parser for command-line usage
    parser = argparse.ArgumentParser(
        description="Convert a folder of 2x speed videos to normal (1x) speed.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'input_folder',
        type=str,
        help="Path to the folder containing the 2x speed videos."
    )
    parser.add_argument(
        'output_folder',
        type=str,
        help="Path to the folder where the normal speed videos will be saved."
    )

    args = parser.parse_args()

    restore_normal_speed(args.input_folder, args.output_folder)

    print("\n--- All videos processed! ---")
