import os
import json
from datetime import datetime
from yt_dlp import YoutubeDL
from moviepy.editor import VideoFileClip
from youtube_comment_downloader import YoutubeCommentDownloader
from pytube import YouTube


def get_video_duration(url):
    try:
        # Suppress noisy output
        ydl_opts = {'quiet': True}
        # ydl_opts = {
        #     'quiet': True,
        #     'cookiefile': 'cookies.txt',  # <--- Add this line
        # }

        # ydl_opts = {
        #     'format': 'bestvideo+bestaudio/best',
        #     # 'outtmpl': os.path.join(video_folder, category, f'{video_id}_original.%(ext)s'),
        #     'postprocessors': [{
        #         'key': 'FFmpegVideoConvertor',
        #         'preferedformat': 'mp4',
        #     }],
        #     'postprocessor_args': [
        #         '-c:v', 'libx264',
        #         '-crf', '23',
        #         '-preset', 'fast',
        #     ],
        #     'merge_output_format': 'mp4',
        #     'cookiesfrombrowser': ('edge', ), 
        #      'cookiefile': 'cookies1.txt', # Change 'chrome' to your browser ('firefox', 'edge', etc.)
        # }



        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)  # Extract metadata
            
            # Extract duration
            duration_seconds = info.get('duration')
            if duration_seconds is None:
                raise ValueError("Could not retrieve duration for this video.")
            
            # Convert duration to hh:mm:ss format
            hours, remainder = divmod(duration_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{hours:02}:{minutes:02}:{seconds:02}"  # Format as hh:mm:ss

    except Exception as e:
        print(f"Error: {e}")
        return None


def get_next_id(json_file):
    """Get the next available ID for the video (sequential numbering)."""
    if os.path.exists(json_file):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return len(data)  # The next available ID is the length of the current data
    return 0  # Start from 0 if the JSON file doesn't exist

def check_video_exists(url, json_file, start_time, end_time):
    """Check if the video URL and time range (within a 30-second boundary) are already in the dataset."""
    def time_to_seconds(time_str):
        """Convert time in HH:MM:SS format to seconds."""
        h, m, s = map(int, time_str.split(":"))
        return h * 3600 + m * 60 + s

    if os.path.exists(json_file):
        start_time_sec = time_to_seconds(start_time)
        end_time_sec = time_to_seconds(end_time)

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

            for entry in data:
                if "url" in entry and entry["url"] == url:
                    entry_start_sec = time_to_seconds(entry["start_time"])
                    entry_end_sec = time_to_seconds(entry["end_time"])

                    # Check if the times fall within a 30-second boundary
                    if (entry_start_sec - 20 <= start_time_sec <= entry_start_sec + 20 and
                            entry_end_sec - 20 <= end_time_sec <= entry_end_sec + 20):
                        return True  # Match found within 30-second boundary

    return False  # Video or time range not found


#delete the video
def delete_file(file_path):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"{file_path} has been deleted successfully.")
        else:
            print(f"The file {file_path} does not exist.")
    except Exception as e:
        print(f"An error occurred while deleting the file: {e}")


# Video trimmer
def trim_video(input_video_path, output_video_path):
    try:
        # Load the video
        video_clip = VideoFileClip(input_video_path, audio=False)  # Disable audio
        trimmed_video = video_clip
        # Write the trimmed video to the output path
        trimmed_video.write_videofile(output_video_path, codec="libx264")
        print(f"Trimmed video saved: {output_video_path}")
        
        # Close the video clip
        video_clip.close()
        trimmed_video.close()
    except Exception as e:
        print(f"An error occurred during trimming: {e}")   


def get_next_video_id(json_file):
    """Get the next video ID based on the highest ID in the JSON file."""
    if not os.path.exists(json_file):
        return 1  # Start with ID 1 if the file doesn't exist

    with open(json_file, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            if not data:  # If the JSON file is empty
                return 1
            # Find the maximum ID in the existing data
            max_id = max(entry.get("id", 0) for entry in data)
            return max_id + 1
        except json.JSONDecodeError:
            return 1  # If JSON is malformed, start with ID 1



def download_video(url, video_folder, audio_folder, comment_folder, video_id):
    try:


        # Setup folder structure
        os.makedirs(video_folder, exist_ok=True)
        os.makedirs(audio_folder, exist_ok=True)
        os.makedirs(comment_folder, exist_ok=True)
        # os.makedirs(mp3_to_text_folder, exist_ok=True)
        # Create subfolders for the category
        os.makedirs(os.path.join(video_folder), exist_ok=True)
        os.makedirs(os.path.join(audio_folder), exist_ok=True)
        os.makedirs(os.path.join(comment_folder), exist_ok=True)
        # os.makedirs(os.path.join(mp3_to_text_folder, category), exist_ok=True)
           
        
        # start_seconds = time_to_seconds(start_time)  
        # end_seconds = time_to_seconds(end_time)  



        # Download full video using yt-dlp
        # ydl_opts = {
        #     #'format': 'best',
        #     'outtmpl': os.path.join(video_folder, f'{video_id}_original.%(ext)s'),
        #     #'listformats': True,  # Add this to list available formats
        # }

        # with YoutubeDL(ydl_opts) as ydl:
        #     info = ydl.extract_info(url, download=True)
        #     video_title = info['title'].replace(" ", "_")
        #     video_path = os.path.join(video_folder, f"{video_id}_original.mp4")
        #     video_description = info.get('description', 'No description available')
        #     print(f"Video downloaded: {video_path}")


  # Download full video using yt-dlp
        ydl_opts = {
            'format': 'bestvideo+bestaudio/best',  # Use best video and audio streams
            'outtmpl': os.path.join(video_folder, f'{video_id}_original.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegVideoConvertor',
                'preferedformat': 'mp4',  # Ensure the video is converted to mp4 format
            }],

            'postprocessor_args': [
                '-c:v', 'libx264',  # Use the x264 codec for video encoding
                '-crf', '23',  # Set the video quality (lower values are better quality, range 0-51)
                '-preset', 'fast',  # Set encoding speed (options: ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow)
            ],
            'merge_output_format': 'mp4',  # Ensure the merged output is in MP4 format
        }

        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_title = info['title'].replace(" ", "_")
            video_path = os.path.join(video_folder, f"{video_id}_original.mp4")
            video_description = info.get('description', 'No description available')
            print(f"Video downloaded: {video_path}")



        
        #trim the video       
        output_path = os.path.join(video_folder, f"{video_id}.mp4")
        trim_video(video_path, output_path)


 
        # Extract audio from the trimmed video for a specific time duration
        audio_path = os.path.join(audio_folder, f"{video_id}.mp3")
        try:
            with VideoFileClip(video_path) as video:
                # Extract only the audio for the given time frame
                audio_clip = video.audio
                audio_clip.write_audiofile(audio_path)
                print(f"Audio extracted for the given time duration: {audio_path}")
                audio_clip.close()  # Close the audio clip
        except Exception as e:
            print(f"Failed to extract audio: {e}")
            audio_path = None

        
        # Delete the original video file
        delete_file(video_path)    

        
        # mp3_to_text_path = os.path.join(mp3_to_text_folder,category, f"{video_id}_mp3totext.json")
        # try:
        # # Create an empty JSON file
        #     with open(mp3_to_text_path, 'w', encoding='utf-8') as f:
        #         json.dump([], f, ensure_ascii=False, indent=4)  # Empty list as initial content
        #         print(f"Empty JSON file created for video ID {video_id}: {mp3_to_text_path}")
        # except Exception as e:
        #     print(f"Error creating JSON file for video ID {video_id}: {e}")

        # Download limited comments
        comment_path = os.path.join(comment_folder, f"{video_id}.json")
        try:
            downloader = YoutubeCommentDownloader()
            comments = downloader.get_comments_from_url(url)
            comments_list = []
            for count, comment in enumerate(comments):
                if count >= 100:  # Limit to 1000 comments
                    break
                comments_list.append(comment['text'])
            with open(comment_path, 'w', encoding="utf-8") as f:
                json.dump(comments_list, f, ensure_ascii=False, indent=4)
            print(f"Comments saved: {comment_path}")
        except Exception as e:
            print(f"Failed to download comments: {e}")
            comment_path = None

        return video_id, video_title, video_description, output_path, audio_path, comment_path,url
      


    except Exception as e:
        print(f"An error occurred: {e}")
        return video_id, None, None, None, None, None, url


def display_categories():
    print("Please select a category:")
    print("1. Baseball bat")
    print("2. Lip kissing")
    print("3. Kill cow with knife")
    print("4. Hit and run")
    print("5. Bomb explotion in public")
    print("6. None")

def get_user_selected_category():
    while True:
        display_categories()
        choice = input("Enter the number of your choice (1-6): ")
        category = ""
        
        if choice == '1':
            category = "Baseball bat"
            break
        elif choice == '2':
            category = "Lip kissing"
            break
        elif choice == '3':
            category = "Kill cow with knife"
            break
        elif choice == '4':
            category = "Hit and run"
            break
        elif choice == '5':
            category = "Bomb explotion in public"
            break        
        elif choice == '6':
            category = "None"
            break
        else:
            print("Invalid choice, please try again.")
    return category,choice




# Function to add data to JSON file
def add_to_json(video_id, video_title, video_description, video_path, audio_path, comment_path,url, json_file):


    time_format = "%H:%M:%S"
    # start = datetime.strptime(start_time, time_format)
    # end = datetime.strptime(end_time, time_format)

    # Calculate the duration as a timedelta object
    # duration = end - start

    # Get the duration in seconds
    # duration_seconds = duration.total_seconds()
    
    data_entry = {
        "id": video_id,
        "video_title": video_title,
        "video_description": video_description,
        "video_path": video_path,
        "audio_path": audio_path,
        "comment_path": comment_path,
        # "mp3_to_text_path" : mp3_to_text_path,
        "url": url,
        # "category": category,
        # "start_time": start_time,
        # "end_time": end_time,
        # "duration": duration_seconds
    }

    # Append data to the JSON file
    if os.path.exists(json_file):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    data.append(data_entry)

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"Data added to {json_file}")

#Time to seconds
def time_to_seconds(timestamp):
    # Split the timestamp into hours, minutes, and seconds
    hours, minutes, seconds = map(int, timestamp.split(":"))
    # Convert hours, minutes, and seconds to total seconds
    total_seconds = hours * 3600 + minutes * 60 + seconds
    return total_seconds


# Main function to execute the process
def main(youtube_url):

    duration = (get_video_duration(youtube_url))
    # isFull = input(f"Duration is {duration}, Do you need to get full video Y/N ? ")

    video_folder = "download/videos"
    audio_folder = "download/audio"
    comment_folder = "download/comments"
    # mp3_to_text_folder = "mp3totext"
    json_file = "download/video_data.json"

    # if isFull in ('N', 'n'):
    #     start_time = input("Enter the Start time eg: HH:mm:ss ?  ")
    #     end_time = input("Enter the End time eg HH:mm:ss: ?  ")

        
    # if isFull in ('Y', 'y'):
    #     start_time = '00:00:00'
    #     end_time = duration


    # Check if the video URL already exists in the JSON file to avoid downloading it again
    # if check_video_exists(youtube_url, json_file, start_time, end_time):
    #     print(f"Video with URL {youtube_url} and same Time stape already exists in the dataset.")
    #     return  # Skip the download if the video already exists
      


    # category,choose = get_user_selected_category()

    # Get the next available ID for the video
    #video_id = get_next_video_id(json_file)

    current_time = datetime.now()

    # Format the system time as "YYYYMMDDHHMMSS"
    formatted_time = current_time.strftime("%Y%m%d%H%M%S")
    video_id = f"{formatted_time}"

    # Download video, extract audio, and save comments
    video_id, video_title, video_description, video_path, audio_path, comment_path, url = download_video(youtube_url, video_folder, audio_folder, comment_folder,video_id)
    return video_id
    # # Save paths and metadata in JSON file
    # if video_id is not None and video_title is not None and video_path is not None and audio_path is not None and comment_path is not None:
    #     add_to_json(video_id, video_title, video_description, video_path, audio_path, comment_path ,url, json_file)
    #     print(f"new audio file- {video_id}")
    #     return video_id
    # else:
    #     print("Could not save all data. Check errors and try again.")

# Run the script
# youtube_url = input("Enter the YouTube video URL: ")
# main(youtube_url)
