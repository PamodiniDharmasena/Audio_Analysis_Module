# # from audio_predictor import predict_audio
# # from video_predictor import predict_video

# # # You can call it with any file path
# # audio_predicted_lable, audio_confident = predict_audio("download/audio/20250329172658.mp3")
# # # print("Final Prediction:", predicted_lable, confident)


# # video_predicted_lable, video_confident = predict_video("download/videos/202503200003375.mp4")


# # print("Prediction:", video_predicted_lable, "Confidence:", round(video_confident, 2))
# # print("Prediction:", audio_predicted_lable, "Confidence:", round(audio_confident, 2))




# import time
# from video_d import main as download_youtube_video
# from audio_predictor import predict_audio
# from video_predictor import predict_video


#     # Step 1: Download YouTube video and get the video_id
# youtube_url = input("Enter the YouTube video URL: ")
# video_id = download_youtube_video(youtube_url)

# # if not video_id:
# #     print("Failed to download the video.")
# #     return
    
# # video_id = '20250510135534' 
# # Step 2: Predict using the downloaded files
# time.sleep(4) 

# audio_path = f"download/audio/{video_id}.mp3"
# video_path = f"download/videos/{video_id}.mp4"

# # Get predictions
# audio_predicted_label, audio_confidence = predict_audio(audio_path)
# video_predicted_label, video_confidence = predict_video(video_path)

# # Print results
# print("\n--- Prediction Results ---")
# print(f"Video Prediction: {video_predicted_label}, Confidence: {round(video_confidence, 2)}")
# print(f"Audio Prediction: {audio_predicted_label}, Confidence: {round(audio_confidence, 2)}")



# import time
# import tkinter as tk
# from tkinter import ttk, messagebox
# from video_d import main as download_youtube_video
# from audio_predictor import predict_audio
# from video_predictor import predict_video
# from comment_predictor import predict_comments


# class VideoAnalyzerApp:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("Video Content Analyzer")
#         self.root.geometry("600x600")
        
#         self.create_widgets()
#         self.setup_layout()
        
#     def create_widgets(self):
#         # Header
#         self.header = ttk.Label(self.root, 
#                               text="YouTube Video Analyzer", 
#                               font=('Arial', 14, 'bold'))
        
#         # URL Entry
#         self.url_frame = ttk.Frame(self.root)
#         self.url_label = ttk.Label(self.url_frame, text="YouTube URL:")
#         self.url_entry = ttk.Entry(self.url_frame, width=50)
        
#         # Progress Frame
#         self.progress_frame = ttk.LabelFrame(self.root, text="Processing Steps")
        
#         # Step 1: Download
#         self.download_frame = ttk.Frame(self.progress_frame)
#         self.download_label = ttk.Label(self.download_frame, text="1. Downloading video...")
#         self.download_progress = ttk.Progressbar(self.download_frame, mode='determinate', length=300)
#         self.download_status = ttk.Label(self.download_frame, text="Waiting", foreground="gray")
        
#         # Step 2: Video Analysis
#         self.video_frame = ttk.Frame(self.progress_frame)
#         self.video_label = ttk.Label(self.video_frame, text="2. Analyzing video content...")
#         self.video_progress = ttk.Progressbar(self.video_frame, mode='determinate', length=300)
#         self.video_status = ttk.Label(self.video_frame, text="Waiting", foreground="gray")
        
#         # Step 3: Audio Analysis
#         self.audio_frame = ttk.Frame(self.progress_frame)
#         self.audio_label = ttk.Label(self.audio_frame, text="3. Analyzing audio content...")
#         self.audio_progress = ttk.Progressbar(self.audio_frame, mode='determinate', length=300)
#         self.audio_status = ttk.Label(self.audio_frame, text="Waiting", foreground="gray")
        
#         # Results
#         self.result_frame = ttk.LabelFrame(self.root, text="Analysis Results")
#         self.result_text = tk.Text(self.result_frame, height=10, wrap=tk.WORD, state='disabled')
#         self.result_scroll = ttk.Scrollbar(self.result_frame, command=self.result_text.yview)
#         self.result_text.config(yscrollcommand=self.result_scroll.set)
        
#         # Control Buttons
#         self.button_frame = ttk.Frame(self.root)
#         self.analyze_button = ttk.Button(self.button_frame, 
#                                         text="Start Analysis", 
#                                         command=self.start_analysis)
#         self.reset_button = ttk.Button(self.button_frame,
#                                      text="Reset",
#                                      command=self.reset_analysis)
        
#     def setup_layout(self):
#         self.header.pack(pady=10)
        
#         # URL Entry
#         self.url_frame.pack(pady=5, padx=10, fill=tk.X)
#         self.url_label.pack(side=tk.LEFT)
#         self.url_entry.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        
#         # Progress Steps
#         self.progress_frame.pack(pady=10, padx=10, fill=tk.X)
        
#         # Download Step
#         self.download_frame.pack(fill=tk.X, padx=5, pady=5)
#         self.download_label.pack(anchor=tk.W)
#         self.download_progress.pack(fill=tk.X, pady=2)
#         self.download_status.pack(anchor=tk.W)
        
#         # Video Analysis Step
#         self.video_frame.pack(fill=tk.X, padx=5, pady=5)
#         self.video_label.pack(anchor=tk.W)
#         self.video_progress.pack(fill=tk.X, pady=2)
#         self.video_status.pack(anchor=tk.W)
        
#         # Audio Analysis Step
#         self.audio_frame.pack(fill=tk.X, padx=5, pady=5)
#         self.audio_label.pack(anchor=tk.W)
#         self.audio_progress.pack(fill=tk.X, pady=2)
#         self.audio_status.pack(anchor=tk.W)
        
#         # Results
#         self.result_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
#         self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
#         self.result_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
#         # Buttons
#         self.button_frame.pack(pady=10)
#         self.analyze_button.pack(side=tk.LEFT, padx=5)
#         self.reset_button.pack(side=tk.LEFT, padx=5)
        
#     def start_analysis(self):
#         youtube_url = self.url_entry.get()
#         if not youtube_url:
#             messagebox.showerror("Error", "Please enter a YouTube URL")
#             return
            
#         self.reset_analysis()
#         self.analyze_button.config(state=tk.DISABLED)
        
#         try:
#             # Step 1: Download video
#             self.update_step(self.download_progress, self.download_status, "Downloading...", 0)
#             self.root.update()
            
#             video_id = download_youtube_video(youtube_url)
#             if not video_id:
#                 raise Exception("Failed to download video")
                
#             for i in range(0, 101, 10):
#                 self.update_step(self.download_progress, self.download_status, f"Downloading... {i}%", i)
#                 self.root.update()
#                 time.sleep(0.1)
                
#             self.update_step(self.download_progress, self.download_status, "Download complete!", 100, "green")
            
#             # Step 2: Video Analysis
#             self.update_step(self.video_progress, self.video_status, "Starting analysis...", 0)
#             self.root.update()
            
#             video_path = f"download/videos/{video_id}.mp4"
#             for i in range(0, 51, 10):
#                 self.update_step(self.video_progress, self.video_status, f"Processing frames... {i}%", i)
#                 self.root.update()
#                 time.sleep(0.1)
                
#             video_predicted_label, video_confidence = predict_video(video_path)
            
#             for i in range(50, 101, 10):
#                 self.update_step(self.video_progress, self.video_status, f"Finalizing... {i}%", i)
#                 self.root.update()
#                 time.sleep(0.05)
                
#             self.update_step(self.video_progress, self.video_status, 
#                            f"Analysis complete: {video_predicted_label}", 100, "green")
            
#             # Step 3: Audio Analysis
#             self.update_step(self.audio_progress, self.audio_status, "Starting analysis...", 0)
#             self.root.update()
            
#             audio_path = f"download/audio/{video_id}.mp3"
#             for i in range(0, 51, 10):
#                 self.update_step(self.audio_progress, self.audio_status, f"Processing audio... {i}%", i)
#                 self.root.update()
#                 time.sleep(0.1)
                
#             audio_predicted_label, audio_confidence = predict_audio(audio_path)
            
#             for i in range(50, 101, 10):
#                 self.update_step(self.audio_progress, self.audio_status, f"Finalizing... {i}%", i)
#                 self.root.update()
#                 time.sleep(0.05)
                
#             self.update_step(self.audio_progress, self.audio_status, 
#                            f"Analysis complete: {audio_predicted_label}", 100, "green")
            

            
#             # Display results
#             self.show_results(video_predicted_label, video_confidence,
#                             audio_predicted_label, audio_confidence)
            
#         except Exception as e:
#             messagebox.showerror("Error", f"Analysis failed: {str(e)}")
#             self.reset_analysis()
#         finally:
#             self.analyze_button.config(state=tk.NORMAL)
            
#     def update_step(self, progress_bar, status_label, text, value, color="black"):
#         progress_bar['value'] = value
#         status_label.config(text=text, foreground=color)
#         self.root.update()
        
#     def show_results(self, video_pred, video_conf, audio_pred, audio_conf):
#         self.result_text.config(state='normal')
#         self.result_text.delete(1.0, tk.END)
        
#         results = "=== ANALYSIS RESULTS ===\n\n"
#         results += f"Video Analysis:\n"
#         results += f"  - Predicted Content: {video_pred}\n"
#         results += f"  - Confidence Level: {round(video_conf, 2)}%\n\n"
#         results += f"Audio Analysis:\n"
#         results += f"  - Predicted Content: {audio_pred}\n"
#         results += f"  - Confidence Level: {round(audio_conf, 2)}%\n\n"
#         results += "Analysis complete!"
        
#         self.result_text.insert(tk.END, results)
#         self.result_text.config(state='disabled')

#     def show_results(self, video_pred, video_conf, audio_pred, audio_conf):
#         label_map_a = {
#             1: 'Baseball Bat Attack',
#             2: 'Bomb Explosion in Public',
#             3: 'Hit and Run',
#             4: 'Kill Cow with Knife',
#             5: 'Lip Kissing',
#             6: 'None'
#         }

#         label_map_v = {
#             '1': 'Baseball Bat Attack',
#             '2': 'Bomb Explosion in Public',
#             '3': 'Hit and Run',
#             '4': 'Kill Cow with Knife',
#             '5': 'Lip Kissing',
#             '6': 'None'
#         }

#         # Convert prediction numbers to descriptive text
#         video_label = label_map_v.get(video_pred, 'Unknown')
#         audio_label = label_map_a.get(audio_pred, 'Unknown')

#         self.result_text.config(state='normal')
#         self.result_text.delete(1.0, tk.END)
        
#         results = "=== ANALYSIS RESULTS ===\n\n"
#         results += f"Video Analysis:\n"
#         results += f"  - Predicted Content: {video_label}\n"
#         results += f"  - Confidence Level: {round(video_conf, 2)}%\n\n"
#         results += f"Audio Analysis:\n"
#         results += f"  - Predicted Content: {audio_label}\n"
#         results += f"  - Confidence Level: {round(audio_conf, 2)}%\n\n"
#         results += "Analysis complete!"
        
#         self.result_text.insert(tk.END, results)
#         self.result_text.config(state='disabled')

        
#     def reset_analysis(self):
#         # Reset progress barsc
#         self.download_progress['value'] = 0
#         self.video_progress['value'] = 0
#         self.audio_progress['value'] = 0
        
#         # Reset status labels
#         self.download_status.config(text="Waiting", foreground="gray")
#         self.video_status.config(text="Waiting", foreground="gray")
#         self.audio_status.config(text="Waiting", foreground="gray")
        
#         # Clear results
#         self.result_text.config(state='normal')
#         self.result_text.delete(1.0, tk.END)
#         self.result_text.insert(tk.END, "Results will appear here after analysis...")
#         self.result_text.config(state='disabled')
        
#         self.root.update()

# if __name__ == "__main__":
#     root = tk.Tk()
#     app = VideoAnalyzerApp(root)
#     root.mainloop()





# import time
# import tkinter as tk
# from tkinter import ttk, messagebox
# from video_d import main as download_youtube_video
# from audio_predictor import predict_audio
# from video_predictor import predict_video
# from comment_predictor import predict_comments  # Newly added import

# class VideoAnalyzerApp:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("Video Content Analyzer")
#         self.root.geometry("600x700")
        
#         self.create_widgets()
#         self.setup_layout()
        
#     def create_widgets(self):
#         self.header = ttk.Label(self.root, text="YouTube Video Analyzer", font=('Arial', 14, 'bold'))
        
#         self.url_frame = ttk.Frame(self.root)
#         self.url_label = ttk.Label(self.url_frame, text="YouTube URL:")
#         self.url_entry = ttk.Entry(self.url_frame, width=50)
        
#         self.progress_frame = ttk.LabelFrame(self.root, text="Processing Steps")
        
#         # Step 1: Download
#         self.download_frame = self.make_progress_section("1. Downloading video...", "Waiting")
        
#         # Step 2: Video Analysis
#         self.video_frame = self.make_progress_section("2. Analyzing video content...", "Waiting")
        
#         # Step 3: Audio Analysis
#         self.audio_frame = self.make_progress_section("3. Analyzing audio content...", "Waiting")

#         # Step 4: Comment Analysis
#         self.comment_frame = self.make_progress_section("4. Analyzing comment content...", "Waiting")

#         self.result_frame = ttk.LabelFrame(self.root, text="Analysis Results")
#         self.result_text = tk.Text(self.result_frame, height=12, wrap=tk.WORD, state='disabled')
#         self.result_scroll = ttk.Scrollbar(self.result_frame, command=self.result_text.yview)
#         self.result_text.config(yscrollcommand=self.result_scroll.set)
        
#         self.button_frame = ttk.Frame(self.root)
#         self.analyze_button = ttk.Button(self.button_frame, text="Start Analysis", command=self.start_analysis)
#         self.reset_button = ttk.Button(self.button_frame, text="Reset", command=self.reset_analysis)

#     def make_progress_section(self, label_text, status_text):
#         frame = ttk.Frame(self.progress_frame)
#         label = ttk.Label(frame, text=label_text)
#         progress = ttk.Progressbar(frame, mode='determinate', length=300)
#         status = ttk.Label(frame, text=status_text, foreground="gray")
#         frame.pack(fill=tk.X, padx=5, pady=5)
#         label.pack(anchor=tk.W)
#         progress.pack(fill=tk.X, pady=2)
#         status.pack(anchor=tk.W)
#         frame.label = label
#         frame.progress = progress
#         frame.status = status
#         return frame

#     def setup_layout(self):
#         self.header.pack(pady=10)
        
#         self.url_frame.pack(pady=5, padx=10, fill=tk.X)
#         self.url_label.pack(side=tk.LEFT)
#         self.url_entry.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        
#         self.progress_frame.pack(pady=10, padx=10, fill=tk.X)
        
#         self.result_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
#         self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
#         self.result_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
#         self.button_frame.pack(pady=10)
#         self.analyze_button.pack(side=tk.LEFT, padx=5)
#         self.reset_button.pack(side=tk.LEFT, padx=5)

#     def start_analysis(self):
#         youtube_url = self.url_entry.get()
#         if not youtube_url:
#             messagebox.showerror("Error", "Please enter a YouTube URL")
#             return
            
#         self.reset_analysis()
#         self.analyze_button.config(state=tk.DISABLED)
        
#         try:
#             self.update_step(self.download_frame, "Downloading...", 0)
#             self.root.update()
            
#             video_id = download_youtube_video(youtube_url)
#             if not video_id:
#                 raise Exception("Failed to download video")
                
#             for i in range(0, 101, 10):
#                 self.update_step(self.download_frame, f"Downloading... {i}%", i)
#                 self.root.update()
#                 time.sleep(0.1)
                
#             self.update_step(self.download_frame, "Download complete!", 100, "green")
            
#             # Video Analysis
#             self.update_step(self.video_frame, "Starting analysis...", 0)
#             self.root.update()
            
#             video_path = f"download/videos/{video_id}.mp4"
#             for i in range(0, 51, 10):
#                 self.update_step(self.video_frame, f"Processing frames... {i}%", i)
#                 self.root.update()
#                 time.sleep(0.1)
                
#             video_predicted_label, video_confidence = predict_video(video_path)
            
#             for i in range(50, 101, 10):
#                 self.update_step(self.video_frame, f"Finalizing... {i}%", i)
#                 self.root.update()
#                 time.sleep(0.05)
                
#             self.update_step(self.video_frame, f"Analysis complete: {video_predicted_label}", 100, "green")
            
#             # Audio Analysis
#             self.update_step(self.audio_frame, "Starting analysis...", 0)
#             self.root.update()
            
#             audio_path = f"download/audio/{video_id}.mp3"
#             for i in range(0, 51, 10):
#                 self.update_step(self.audio_frame, f"Processing audio... {i}%", i)
#                 self.root.update()
#                 time.sleep(0.1)
                
#             audio_predicted_label, audio_confidence = predict_audio(audio_path)
            
#             for i in range(50, 101, 10):
#                 self.update_step(self.audio_frame, f"Finalizing... {i}%", i)
#                 self.root.update()
#                 time.sleep(0.05)
                
#             self.update_step(self.audio_frame, f"Analysis complete: {audio_predicted_label}", 100, "green")

#             # Comment Analysis
#             self.update_step(self.comment_frame, "Starting analysis...", 0)
#             self.root.update()

#             comment_path = f"download/comments/{video_id}.json"
#             for i in range(0, 51, 10):
#                 self.update_step(self.comment_frame, f"Processing comments... {i}%", i)
#                 self.root.update()
#                 time.sleep(0.1)

#             comment_predicted_label, comment_confidence = predict_comments(comment_path)

#             for i in range(50, 101, 10):
#                 self.update_step(self.comment_frame, f"Finalizing... {i}%", i)
#                 self.root.update()
#                 time.sleep(0.05)

#             self.update_step(self.comment_frame, f"Analysis complete: {comment_predicted_label}", 100, "green")

#             # Display final results
#             self.show_results(video_predicted_label, video_confidence,
#                               audio_predicted_label, audio_confidence,
#                               comment_predicted_label, comment_confidence)
            
#         except Exception as e:
#             messagebox.showerror("Error", f"Analysis failed: {str(e)}")
#             self.reset_analysis()
#         finally:
#             self.analyze_button.config(state=tk.NORMAL)

#     def update_step(self, frame, text, value, color="black"):
#         frame.progress['value'] = value
#         frame.status.config(text=text, foreground=color)
#         self.root.update()

#     def show_results(self, video_pred, video_conf, audio_pred, audio_conf, comment_pred, comment_conf):
#         # label_map_a = {
#         #     1: 'Baseball Bat Attack',
#         #     2: 'Bomb Explosion in Public',
#         #     3: 'Hit and Run',
#         #     4: 'Kill Cow with Knife',
#         #     5: 'Lip Kissing',
#         #     6: 'None'
#         # }

#         label_map = {
#             '1': 'Baseball Bat Attack',
#             '2': 'Bomb Explosion in Public',
#             '3': 'Hit and Run',
#             '4': 'Kill Cow with Knife',
#             '5': 'Lip Kissing',
#             '6': 'None'
#         }

  

#         video_label = label_map.get(str(video_pred), 'Unknown')
#         audio_label = label_map.get(str(audio_pred), 'Unknown')
#         comment_label = label_map.get(str(comment_pred), 'Unknown') 
#         # label_map.get(str(comment_pred), 'Unknown')

#         self.result_text.config(state='normal')
#         self.result_text.delete(1.0, tk.END)
        
#         results = "=== ANALYSIS RESULTS ===\n\n"
#         results += f"Video Analysis:\n"
#         results += f"  - Predicted Content: {video_label}\n"
#         results += f"  - Confidence Level: {round(video_conf, 2)}%\n\n"
#         results += f"Audio Analysis:\n"
#         results += f"  - Predicted Content: {audio_label}\n"
#         results += f"  - Confidence Level: {round(audio_conf, 2)}%\n\n"
#         results += f"Comment Analysis:\n"
#         results += f"  - Predicted Content: {comment_label}\n"
#         results += f"  - Confidence Level: {round(comment_conf, 2)}%\n\n"
#         results += "Analysis complete!"
        
#         self.result_text.insert(tk.END, results)
#         self.result_text.config(state='disabled')

#     def reset_analysis(self):
#         for frame in [self.download_frame, self.video_frame, self.audio_frame, self.comment_frame]:
#             frame.progress['value'] = 0
#             frame.status.config(text="Waiting", foreground="gray")
        
#         self.result_text.config(state='normal')
#         self.result_text.delete(1.0, tk.END)
#         self.result_text.insert(tk.END, "Results will appear here after analysis...")
#         self.result_text.config(state='disabled')
#         self.root.update()

# if __name__ == "__main__":
#     root = tk.Tk()
#     app = VideoAnalyzerApp(root)
#     root.mainloop()




# import time
# import tkinter as tk
# from tkinter import ttk, messagebox
# from video_d import main as download_youtube_video
# from audio_predictor import predict_audio
# from video_predictor import predict_video
# from comment_predictor import predict_comments
# import joblib
# import pandas as pd

# class VideoAnalyzerApp:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("Video Content Analyzer")
#         self.root.geometry("600x650")  # Optimized window size
        
#         # Load KNN model
#         try:
#             self.knn_model = joblib.load("knn_model.joblib")
#         except Exception as e:
#             messagebox.showerror("Error", f"Failed to load KNN model: {str(e)}")
#             self.knn_model = None
        
#         self.create_widgets()
#         self.setup_layout()
        
#     def create_widgets(self):
#         # Header
#         self.header = ttk.Label(self.root, text="YouTube Video Analyzer", font=('Arial', 14, 'bold'))
        
#         # URL Entry
#         self.url_frame = ttk.Frame(self.root)
#         self.url_label = ttk.Label(self.url_frame, text="YouTube URL:")
#         self.url_entry = ttk.Entry(self.url_frame, width=40)
        
#         # Progress Tracking
#         self.progress_frame = ttk.LabelFrame(self.root, text="Processing Steps", padding=5)
#         self.download_frame = self.make_progress_section("1. Downloading video...", "Waiting")
#         self.video_frame = self.make_progress_section("2. Analyzing video content...", "Waiting")
#         self.audio_frame = self.make_progress_section("3. Analyzing audio content...", "Waiting")
#         self.comment_frame = self.make_progress_section("4. Analyzing comment content...", "Waiting")

#         # Notebook for tabs
#         self.notebook = ttk.Notebook(self.root)
        
#         # Analysis Results Tab
#         self.results_tab = ttk.Frame(self.notebook)
#         self.result_text = tk.Text(self.results_tab, height=8, wrap=tk.WORD, 
#                                  state='disabled', padx=5, pady=5)
#         self.result_scroll = ttk.Scrollbar(self.results_tab, command=self.result_text.yview)
#         self.result_text.config(yscrollcommand=self.result_scroll.set)
        
#         # KNN Results Tab
#         self.knn_tab = ttk.Frame(self.notebook)
#         self.knn_text = tk.Text(self.knn_tab, height=4, wrap=tk.WORD, 
#                               state='disabled', padx=5, pady=5)
        
#         # Buttons
#         self.button_frame = ttk.Frame(self.root)
#         self.analyze_button = ttk.Button(self.button_frame, text="Start Analysis", 
#                                        command=self.start_analysis, width=12)
#         self.knn_button = ttk.Button(self.button_frame, text="KNN Analysis", 
#                                    command=self.run_knn_analysis, state=tk.DISABLED, width=12)
#         self.reset_button = ttk.Button(self.button_frame, text="Reset", 
#                                      command=self.reset_analysis, width=12)

#     def make_progress_section(self, label_text, status_text):
#         frame = ttk.Frame(self.progress_frame, padding=(0, 2, 0, 2))
#         label = ttk.Label(frame, text=label_text)
#         progress = ttk.Progressbar(frame, mode='determinate', length=250)
#         status = ttk.Label(frame, text=status_text, foreground="gray")
#         frame.pack(fill=tk.X, padx=2, pady=1)
#         label.pack(anchor=tk.W)
#         progress.pack(fill=tk.X, pady=1)
#         status.pack(anchor=tk.W)
#         frame.label = label
#         frame.progress = progress
#         frame.status = status
#         return frame

#     def setup_layout(self):
#         self.header.pack(pady=5)
#         self.url_frame.pack(pady=2, padx=5, fill=tk.X)
#         self.url_label.pack(side=tk.LEFT)
#         self.url_entry.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
#         self.progress_frame.pack(pady=5, padx=5, fill=tk.X)
        
#         # Notebook setup
#         self.notebook.add(self.results_tab, text="Analysis Results")
#         self.notebook.add(self.knn_tab, text="KNN Prediction")
#         self.notebook.pack(pady=5, padx=5, fill=tk.BOTH, expand=True)
        
#         # Results tab layout
#         self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
#         self.result_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
#         # KNN tab layout
#         self.knn_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
#         # Buttons layout
#         self.button_frame.pack(pady=5)
#         self.analyze_button.pack(side=tk.LEFT, padx=2)
#         self.knn_button.pack(side=tk.LEFT, padx=2)
#         self.reset_button.pack(side=tk.LEFT, padx=2)

#     def start_analysis(self):
#         youtube_url = self.url_entry.get()
#         if not youtube_url:
#             messagebox.showerror("Error", "Please enter a YouTube URL")
#             return
            
#         self.reset_analysis()
#         self.analyze_button.config(state=tk.DISABLED)
#         self.knn_button.config(state=tk.DISABLED)
        
#         try:
#             # Download video
#             self.update_step(self.download_frame, "Downloading...", 0)
#             video_id = download_youtube_video(youtube_url)
#             if not video_id:
#                 raise Exception("Failed to download video")
                
#             for i in range(0, 101, 10):
#                 self.update_step(self.download_frame, f"Downloading... {i}%", i)
#                 self.root.update()
#                 time.sleep(0.1)
#             self.update_step(self.download_frame, "Download complete!", 100, "green")
            
#             # Video Analysis
#             self.update_step(self.video_frame, "Starting analysis...", 0)
#             video_path = f"download/videos/{video_id}.mp4"
#             for i in range(0, 51, 10):
#                 self.update_step(self.video_frame, f"Processing frames... {i}%", i)
#                 self.root.update()
#                 time.sleep(0.1)
                
#             video_pred, video_conf = predict_video(video_path)
#             for i in range(50, 101, 10):
#                 self.update_step(self.video_frame, f"Finalizing... {i}%", i)
#                 self.root.update()
#                 time.sleep(0.05)
#             self.update_step(self.video_frame, f"Analysis complete", 100, "green")
            
#             # Audio Analysis
#             self.update_step(self.audio_frame, "Starting analysis...", 0)
#             audio_path = f"download/audio/{video_id}.mp3"
#             for i in range(0, 51, 10):
#                 self.update_step(self.audio_frame, f"Processing audio... {i}%", i)
#                 self.root.update()
#                 time.sleep(0.1)
                
#             audio_pred, audio_conf = predict_audio(audio_path)
#             for i in range(50, 101, 10):
#                 self.update_step(self.audio_frame, f"Finalizing... {i}%", i)
#                 self.root.update()
#                 time.sleep(0.05)
#             self.update_step(self.audio_frame, f"Analysis complete", 100, "green")

#             # Comment Analysis
#             self.update_step(self.comment_frame, "Starting analysis...", 0)
#             comment_path = f"download/comments/{video_id}.json"
#             for i in range(0, 51, 10):
#                 self.update_step(self.comment_frame, f"Processing comments... {i}%", i)
#                 self.root.update()
#                 time.sleep(0.1)

#             comment_pred, comment_conf = predict_comments(comment_path)
#             for i in range(50, 101, 10):
#                 self.update_step(self.comment_frame, f"Finalizing... {i}%", i)
#                 self.root.update()
#                 time.sleep(0.05)
#             self.update_step(self.comment_frame, f"Analysis complete", 100, "green")

#             # Store results for KNN analysis
#             self.current_results = {
#                 'video_pred': video_pred,
#                 'video_conf': video_conf,
#                 'audio_pred': audio_pred,
#                 'audio_conf': audio_conf,
#                 'comment_pred': comment_pred,
#                 'comment_conf': comment_conf
#             }

#             # Display results
#             self.show_results(video_pred, video_conf, audio_pred, audio_conf, 
#                             comment_pred, comment_conf)
            
#             # Enable KNN button if model is loaded
#             if self.knn_model is not None:
#                 self.knn_button.config(state=tk.NORMAL)
            
#         except Exception as e:
#             messagebox.showerror("Error", f"Analysis failed: {str(e)}")
#             self.reset_analysis()
#         finally:
#             self.analyze_button.config(state=tk.NORMAL)

#     def run_knn_analysis(self):
#         if not hasattr(self, 'current_results'):
#             messagebox.showerror("Error", "Please run analysis first")
#             return
            
#         if self.knn_model is None:
#             messagebox.showerror("Error", "KNN model not loaded")
#             return
            
#         try:
#             # Prepare features for KNN
#             features = pd.DataFrame({
#                 'video_prediction': [self.current_results['video_pred']],
#                 'video_confidence': [self.current_results['video_conf']],
#                 'audio_prediction': [self.current_results['audio_pred']],
#                 'audio_confidence': [self.current_results['audio_conf']],
#                 'text_prediction': [self.current_results['comment_pred']],
#                 'text_confidence': [self.current_results['comment_conf']]
#             })
            
#             # Convert all features to numeric
#             features = features.apply(pd.to_numeric, errors='coerce').fillna(0)
            
#             # Make prediction
#             knn_prediction = self.knn_model['model'].predict(features)
#             predicted_category = self.knn_model['label_encoder'].inverse_transform(knn_prediction)[0]
            
#             # Get category name
#             label_map = {
#                 '1': 'Baseball Bat Attack',
#                 '2': 'Bomb Explosion in Public',
#                 '3': 'Hit and Run',
#                 '4': 'Kill Cow with Knife',
#                 '5': 'Lip Kissing',
#                 '6': 'None'
#             }
            
#             predicted_label = label_map.get(str(predicted_category), 'Unknown')
            
#             # Display KNN results
#             self.knn_text.config(state='normal')
#             self.knn_text.delete(1.0, tk.END)
            
#             knn_results = "=== KNN FINAL PREDICTION ===\n\n"
#             knn_results += f"Based on combined analysis of all modalities:\n"
#             knn_results += f"  - Predicted Content: {predicted_label}\n"
#             knn_results += f"  - Predicted Category: {predicted_category}\n\n"
            
#             self.knn_text.insert(tk.END, knn_results)
#             self.knn_text.config(state='disabled')
            
#             # Switch to KNN tab
#             self.notebook.select(self.knn_tab)
            
#         except Exception as e:
#             messagebox.showerror("Error", f"KNN analysis failed: {str(e)}")

#     def update_step(self, frame, text, value, color="black"):
#         frame.progress['value'] = value
#         frame.status.config(text=text, foreground=color)
#         self.root.update()

#     def show_results(self, video_pred, video_conf, audio_pred, audio_conf, comment_pred, comment_conf):
#         label_map = {
#             '1': 'Baseball Bat Attack',
#             '2': 'Bomb Explosion in Public',
#             '3': 'Hit and Run',
#             '4': 'Kill Cow with Knife',
#             '5': 'Lip Kissing',
#             '6': 'None'
#         }

#         video_label = label_map.get(str(video_pred), 'Unknown')
#         audio_label = label_map.get(str(audio_pred), 'Unknown')
#         comment_label = label_map.get(str(comment_pred), 'Unknown')

#         self.result_text.config(state='normal')
#         self.result_text.delete(1.0, tk.END)
        
#         results = "=== ANALYSIS RESULTS ===\n\n"
#         results += f"Video Analysis:\n"
#         results += f"  - Content: {video_label}\n"
#         results += f"  - Confidence: {round(video_conf, 2)}%\n\n"
#         results += f"Audio Analysis:\n"
#         results += f"  - Content: {audio_label}\n"
#         results += f"  - Confidence: {round(audio_conf, 2)}%\n\n"
#         results += f"Comment Analysis:\n"
#         results += f"  - Content: {comment_label}\n"
#         results += f"  - Confidence: {round(comment_conf, 2)}%\n\n"
#         results += "Click 'KNN Analysis' for final prediction"
        
#         self.result_text.insert(tk.END, results)
#         self.result_text.config(state='disabled')

#     def reset_analysis(self):
#         for frame in [self.download_frame, self.video_frame, self.audio_frame, self.comment_frame]:
#             frame.progress['value'] = 0
#             frame.status.config(text="Waiting", foreground="gray")
        
#         self.result_text.config(state='normal')
#         self.result_text.delete(1.0, tk.END)
#         self.result_text.insert(tk.END, "Results will appear here after analysis...")
#         self.result_text.config(state='disabled')
        
#         self.knn_text.config(state='normal')
#         self.knn_text.delete(1.0, tk.END)
#         self.knn_text.insert(tk.END, "Run analysis first, then click 'KNN Analysis'")
#         self.knn_text.config(state='disabled')
        
#         self.knn_button.config(state=tk.DISABLED)
#         self.notebook.select(self.results_tab)
        
#         if hasattr(self, 'current_results'):
#             del self.current_results
            
#         self.root.update()

# if __name__ == "__main__":
#     root = tk.Tk()
#     app = VideoAnalyzerApp(root)
#     root.mainloop()










# import time
# import tkinter as tk
# from tkinter import ttk, messagebox
# from video_d import main as download_youtube_video
# from audio_predictor import predict_audio
# from video_predictor import predict_video
# from comment_predictor import predict_comments
# import joblib
# import pandas as pd
# import numpy as np

# class VideoAnalyzerApp:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("Video Content Analyzer")
#         self.root.geometry("600x700")  # Slightly taller window for additional tab
        
#         # Load KNN model
#         try:
#             self.knn_model = joblib.load("knn_model.joblib")
#         except Exception as e:
#             messagebox.showerror("Error", f"Failed to load KNN model: {str(e)}")
#             self.knn_model = None
        
#         self.create_widgets()
#         self.setup_layout()
        
#     def create_widgets(self):
#         # Header
#         self.header = ttk.Label(self.root, text="YouTube Video Analyzer", font=('Arial', 14, 'bold'))
        
#         # URL Entry
#         self.url_frame = ttk.Frame(self.root)
#         self.url_label = ttk.Label(self.url_frame, text="YouTube URL:")
#         self.url_entry = ttk.Entry(self.url_frame, width=40)
        
#         # Progress Tracking
#         self.progress_frame = ttk.LabelFrame(self.root, text="Processing Steps", padding=5)
#         self.download_frame = self.make_progress_section("1. Downloading video...", "Waiting")
#         self.video_frame = self.make_progress_section("2. Analyzing video content...", "Waiting")
#         self.audio_frame = self.make_progress_section("3. Analyzing audio content...", "Waiting")
#         self.comment_frame = self.make_progress_section("4. Analyzing comment content...", "Waiting")

#         # Notebook for tabs
#         self.notebook = ttk.Notebook(self.root)
        
#         # Analysis Results Tab
#         self.results_tab = ttk.Frame(self.notebook)
#         self.result_text = tk.Text(self.results_tab, height=8, wrap=tk.WORD, 
#                                  state='disabled', padx=5, pady=5)
#         self.result_scroll = ttk.Scrollbar(self.results_tab, command=self.result_text.yview)
#         self.result_text.config(yscrollcommand=self.result_scroll.set)
        
#         # KNN Results Tab
#         self.knn_tab = ttk.Frame(self.notebook)
#         self.knn_text = tk.Text(self.knn_tab, height=4, wrap=tk.WORD, 
#                               state='disabled', padx=5, pady=5)
        
#         # Weighted Vote Results Tab
#         self.weighted_tab = ttk.Frame(self.notebook)
#         self.weighted_text = tk.Text(self.weighted_tab, height=4, wrap=tk.WORD,
#                                   state='disabled', padx=5, pady=5)
        
#         # Buttons
#         self.button_frame = ttk.Frame(self.root)
#         self.analyze_button = ttk.Button(self.button_frame, text="Start Analysis", 
#                                        command=self.start_analysis, width=12)
#         self.knn_button = ttk.Button(self.button_frame, text="KNN Analysis", 
#                                    command=self.run_knn_analysis, state=tk.DISABLED, width=12)
#         self.weighted_button = ttk.Button(self.button_frame, text="Weighted Vote", 
#                                        command=self.run_weighted_vote, state=tk.DISABLED, width=12)
#         self.reset_button = ttk.Button(self.button_frame, text="Reset", 
#                                      command=self.reset_analysis, width=12)

#     def make_progress_section(self, label_text, status_text):
#         frame = ttk.Frame(self.progress_frame, padding=(0, 2, 0, 2))
#         label = ttk.Label(frame, text=label_text)
#         progress = ttk.Progressbar(frame, mode='determinate', length=250)
#         status = ttk.Label(frame, text=status_text, foreground="gray")
#         frame.pack(fill=tk.X, padx=2, pady=1)
#         label.pack(anchor=tk.W)
#         progress.pack(fill=tk.X, pady=1)
#         status.pack(anchor=tk.W)
#         frame.label = label
#         frame.progress = progress
#         frame.status = status
#         return frame

#     def setup_layout(self):
#         self.header.pack(pady=5)
#         self.url_frame.pack(pady=2, padx=5, fill=tk.X)
#         self.url_label.pack(side=tk.LEFT)
#         self.url_entry.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
#         self.progress_frame.pack(pady=5, padx=5, fill=tk.X)
        
#         # Notebook setup
#         self.notebook.add(self.results_tab, text="Analysis Results")
#         self.notebook.add(self.knn_tab, text="KNN Prediction")
#         self.notebook.add(self.weighted_tab, text="Weighted Vote")
#         self.notebook.pack(pady=5, padx=5, fill=tk.BOTH, expand=True)
        
#         # Results tab layout
#         self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
#         self.result_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
#         # KNN tab layout
#         self.knn_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
#         # Weighted tab layout
#         self.weighted_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
#         # Buttons layout
#         self.button_frame.pack(pady=5)
#         self.analyze_button.pack(side=tk.LEFT, padx=2)
#         self.knn_button.pack(side=tk.LEFT, padx=2)
#         self.weighted_button.pack(side=tk.LEFT, padx=2)
#         self.reset_button.pack(side=tk.LEFT, padx=2)

#     def start_analysis(self):
#         youtube_url = self.url_entry.get()
#         if not youtube_url:
#             messagebox.showerror("Error", "Please enter a YouTube URL")
#             return
            
#         self.reset_analysis()
#         self.analyze_button.config(state=tk.DISABLED)
#         self.knn_button.config(state=tk.DISABLED)
#         self.weighted_button.config(state=tk.DISABLED)
        
#         try:
#             # Download video
#             self.update_step(self.download_frame, "Downloading...", 0)
#             video_id = download_youtube_video(youtube_url)
#             if not video_id:
#                 raise Exception("Failed to download video")
                
#             for i in range(0, 101, 10):
#                 self.update_step(self.download_frame, f"Downloading... {i}%", i)
#                 self.root.update()
#                 time.sleep(0.1)
#             self.update_step(self.download_frame, "Download complete!", 100, "green")
            
#             # Video Analysis
#             self.update_step(self.video_frame, "Starting analysis...", 0)
#             video_path = f"download/videos/{video_id}.mp4"
#             for i in range(0, 51, 10):
#                 self.update_step(self.video_frame, f"Processing frames... {i}%", i)
#                 self.root.update()
#                 time.sleep(0.1)
                
#             video_pred, video_conf = predict_video(video_path)
#             for i in range(50, 101, 10):
#                 self.update_step(self.video_frame, f"Finalizing... {i}%", i)
#                 self.root.update()
#                 time.sleep(0.05)
#             self.update_step(self.video_frame, f"Analysis complete", 100, "green")
            
#             # Audio Analysis
#             self.update_step(self.audio_frame, "Starting analysis...", 0)
#             audio_path = f"download/audio/{video_id}.mp3"
#             for i in range(0, 51, 10):
#                 self.update_step(self.audio_frame, f"Processing audio... {i}%", i)
#                 self.root.update()
#                 time.sleep(0.1)
                
#             audio_pred, audio_conf = predict_audio(audio_path)
#             for i in range(50, 101, 10):
#                 self.update_step(self.audio_frame, f"Finalizing... {i}%", i)
#                 self.root.update()
#                 time.sleep(0.05)
#             self.update_step(self.audio_frame, f"Analysis complete", 100, "green")

#             # Comment Analysis
#             self.update_step(self.comment_frame, "Starting analysis...", 0)
#             comment_path = f"download/comments/{video_id}.json"
#             for i in range(0, 51, 10):
#                 self.update_step(self.comment_frame, f"Processing comments... {i}%", i)
#                 self.root.update()
#                 time.sleep(0.1)

#             comment_pred, comment_conf = predict_comments(comment_path)
#             for i in range(50, 101, 10):
#                 self.update_step(self.comment_frame, f"Finalizing... {i}%", i)
#                 self.root.update()
#                 time.sleep(0.05)
#             self.update_step(self.comment_frame, f"Analysis complete", 100, "green")

#             # Store results for KNN and weighted vote analysis
#             self.current_results = {
#                 'video_pred': video_pred,
#                 'video_conf': video_conf,
#                 'audio_pred': audio_pred,
#                 'audio_conf': audio_conf,
#                 'comment_pred': comment_pred,
#                 'comment_conf': comment_conf
#             }

#             # Display results
#             self.show_results(video_pred, video_conf, audio_pred, audio_conf, 
#                             comment_pred, comment_conf)
            
#             # Enable analysis buttons
#             self.knn_button.config(state=tk.NORMAL)
#             self.weighted_button.config(state=tk.NORMAL)
            
#         except Exception as e:
#             messagebox.showerror("Error", f"Analysis failed: {str(e)}")
#             self.reset_analysis()
#         finally:
#             self.analyze_button.config(state=tk.NORMAL)

#     def run_knn_analysis(self):
#         if not hasattr(self, 'current_results'):
#             messagebox.showerror("Error", "Please run analysis first")
#             return
            
#         if self.knn_model is None:
#             messagebox.showerror("Error", "KNN model not loaded")
#             return
            
#         try:
#             # Prepare features for KNN
#             features = pd.DataFrame({
#                 'video_prediction': [self.current_results['video_pred']],
#                 'video_confidence': [self.current_results['video_conf']],
#                 'audio_prediction': [self.current_results['audio_pred']],
#                 'audio_confidence': [self.current_results['audio_conf']],
#                 'text_prediction': [self.current_results['comment_pred']],
#                 'text_confidence': [self.current_results['comment_conf']]
#             })
            
#             # Convert all features to numeric
#             features = features.apply(pd.to_numeric, errors='coerce').fillna(0)
            
#             # Make prediction
#             knn_prediction = self.knn_model['model'].predict(features)
#             predicted_category = self.knn_model['label_encoder'].inverse_transform(knn_prediction)[0]
            
#             # Get category name
#             predicted_label = self.get_category_label(predicted_category)
            
#             # Display KNN results
#             self.knn_text.config(state='normal')
#             self.knn_text.delete(1.0, tk.END)
            
#             knn_results = "=== KNN FINAL PREDICTION ===\n\n"
#             knn_results += f"Based on combined analysis of all modalities:\n"
#             knn_results += f"  - Predicted Content: {predicted_label}\n"
#             knn_results += f"  - Predicted Category: {predicted_category}\n\n"
#             knn_results += "Note: KNN prediction considers all modalities with learned weights"
            
#             self.knn_text.insert(tk.END, knn_results)
#             self.knn_text.config(state='disabled')
            
#             # Switch to KNN tab
#             self.notebook.select(self.knn_tab)
            
#         except Exception as e:
#             messagebox.showerror("Error", f"KNN analysis failed: {str(e)}")

#     def run_weighted_vote(self):
#         if not hasattr(self, 'current_results'):
#             messagebox.showerror("Error", "Please run analysis first")
#             return
            
#         try:
#             # Get predictions and confidences
#             video_pred = int(self.current_results['video_pred'])
#             video_conf = self.current_results['video_conf'] / 100  # Convert to 0-1 scale
#             audio_pred = int(self.current_results['audio_pred'])
#             audio_conf = self.current_results['audio_conf'] / 100
#             comment_pred = int(self.current_results['comment_pred'])
#             comment_conf = self.current_results['comment_conf'] / 100
            
#             # Define weights for each modality (can be adjusted)
#             weights = {
#                 'video': 0.5,  # Highest weight for video
#                 'audio': 0.3,
#                 'comment': 0.2
#             }
            
#             # Create weighted votes
#             weighted_votes = {
#                 1: 0,  # Baseball Bat Attack
#                 2: 0,  # Bomb Explosion in Public
#                 3: 0,  # Hit and Run
#                 4: 0,  # Kill Cow with Knife
#                 5: 0,  # Lip Kissing
#                 6: 0   # None
#             }
            
#             # Add weighted votes from each modality
#             weighted_votes[video_pred] += weights['video'] * video_conf
#             weighted_votes[audio_pred] += weights['audio'] * audio_conf
#             weighted_votes[comment_pred] += weights['comment'] * comment_conf
            
#             # Get the prediction with highest weighted vote
#             final_prediction = max(weighted_votes.items(), key=lambda x: x[1])[0]
#             final_confidence = weighted_votes[final_prediction] * 100  # Convert back to percentage
            
#             # Get category name
#             predicted_label = self.get_category_label(final_prediction)
            
#             # Display weighted vote results
#             self.weighted_text.config(state='normal')
#             self.weighted_text.delete(1.0, tk.END)
            
#             weighted_results = "=== WEIGHTED VOTE PREDICTION ===\n\n"
#             weighted_results += f"Based on weighted combination of modalities:\n"
#             weighted_results += f"  - Predicted Content: {predicted_label}\n"
#             weighted_results += f"  - Predicted Category: {final_prediction}\n"
#             weighted_results += f"  - Combined Confidence: {round(final_confidence, 2)}%\n\n"
#             weighted_results += "Weights used:\n"
#             weighted_results += f"  - Video: {weights['video']*100}%\n"
#             weighted_results += f"  - Audio: {weights['audio']*100}%\n"
#             weighted_results += f"  - Comments: {weights['comment']*100}%"
            
#             self.weighted_text.insert(tk.END, weighted_results)
#             self.weighted_text.config(state='disabled')
            
#             # Switch to weighted vote tab
#             self.notebook.select(self.weighted_tab)
            
#         except Exception as e:
#             messagebox.showerror("Error", f"Weighted vote analysis failed: {str(e)}")

#     def get_category_label(self, category):
#         label_map = {
#             1: 'Baseball Bat Attack',
#             2: 'Bomb Explosion in Public',
#             3: 'Hit and Run',
#             4: 'Kill Animals',
#             5: 'Heavy Lip Kissing',
#             6: 'None'
#         }
#         return label_map.get(category, 'Unknown')

#     def update_step(self, frame, text, value, color="black"):
#         frame.progress['value'] = value
#         frame.status.config(text=text, foreground=color)
#         self.root.update()

#     def show_results(self, video_pred, video_conf, audio_pred, audio_conf, comment_pred, comment_conf):
#         video_label = self.get_category_label(int(video_pred))
#         audio_label = self.get_category_label(int(audio_pred))
#         comment_label = self.get_category_label(int(comment_pred))

#         self.result_text.config(state='normal')
#         self.result_text.delete(1.0, tk.END)
        
#         results = "=== ANALYSIS RESULTS ===\n\n"
#         results += f"Video Analysis:\n"
#         results += f"  - Content: {video_label}\n"
#         results += f"  - Confidence: {round(video_conf, 2)}%\n\n"
#         results += f"Audio Analysis:\n"
#         results += f"  - Content: {audio_label}\n"
#         results += f"  - Confidence: {round(audio_conf, 2)}%\n\n"
#         results += f"Comment Analysis:\n"
#         results += f"  - Content: {comment_label}\n"
#         results += f"  - Confidence: {round(comment_conf, 2)}%\n\n"
#         results += "Click 'KNN Analysis' or 'Weighted Vote' for final prediction"
        
#         self.result_text.insert(tk.END, results)
#         self.result_text.config(state='disabled')

#     def reset_analysis(self):
#         for frame in [self.download_frame, self.video_frame, self.audio_frame, self.comment_frame]:
#             frame.progress['value'] = 0
#             frame.status.config(text="Waiting", foreground="gray")
        
#         self.result_text.config(state='normal')
#         self.result_text.delete(1.0, tk.END)
#         self.result_text.insert(tk.END, "Results will appear here after analysis...")
#         self.result_text.config(state='disabled')
        
#         self.knn_text.config(state='normal')
#         self.knn_text.delete(1.0, tk.END)
#         self.knn_text.insert(tk.END, "Run analysis first, then click 'KNN Analysis'")
#         self.knn_text.config(state='disabled')
        
#         self.weighted_text.config(state='normal')
#         self.weighted_text.delete(1.0, tk.END)
#         self.weighted_text.insert(tk.END, "Run analysis first, then click 'Weighted Vote'")
#         self.weighted_text.config(state='disabled')
        
#         self.knn_button.config(state=tk.DISABLED)
#         self.weighted_button.config(state=tk.DISABLED)
#         self.notebook.select(self.results_tab)
        
#         if hasattr(self, 'current_results'):
#             del self.current_results
            
#         self.root.update()

# if __name__ == "__main__":
#     root = tk.Tk()
#     app = VideoAnalyzerApp(root)
#     root.mainloop()






















import time
import tkinter as tk
from tkinter import ttk, messagebox
from video_d import main as download_youtube_video
from audio_predictor import predict_audio
from video_predictor import predict_video
from comment_predictor import predict_comments
import joblib
import pandas as pd
import numpy as np

class VideoAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Content Analyzer")
        self.root.geometry("600x700")
        
        # Load KNN model
        try:
            self.knn_model = joblib.load("knn_model.joblib")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load KNN model: {str(e)}")
            self.knn_model = None
        
        self.create_widgets()
        self.setup_layout()
        
    def create_widgets(self):
        # Header
        self.header = ttk.Label(self.root, text="YouTube Video Analyzer", font=('Arial', 14, 'bold'))
        
        # URL Entry
        self.url_frame = ttk.Frame(self.root)
        self.url_label = ttk.Label(self.url_frame, text="YouTube URL:")
        self.url_entry = ttk.Entry(self.url_frame, width=40)
        
        # Progress Tracking
        self.progress_frame = ttk.LabelFrame(self.root, text="Processing Steps", padding=5)
        self.download_frame = self.make_progress_section("1. Downloading video...", "Waiting")
        self.video_frame = self.make_progress_section("2. Analyzing video content...", "Waiting")
        self.audio_frame = self.make_progress_section("3. Analyzing audio content...", "Waiting")
        self.comment_frame = self.make_progress_section("4. Analyzing comment content...", "Waiting")

        # Notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        
        # Analysis Results Tab
        self.results_tab = ttk.Frame(self.notebook)
        self.result_text = tk.Text(self.results_tab, height=8, wrap=tk.WORD, 
                                 state='disabled', padx=5, pady=5)
        self.result_scroll = ttk.Scrollbar(self.results_tab, command=self.result_text.yview)
        self.result_text.config(yscrollcommand=self.result_scroll.set)
        
        # KNN Results Tab
        self.knn_tab = ttk.Frame(self.notebook)
        self.knn_text = tk.Text(self.knn_tab, height=4, wrap=tk.WORD, 
                              state='disabled', padx=5, pady=5)
        
        # Weighted Vote Results Tab
        self.weighted_tab = ttk.Frame(self.notebook)
        self.weighted_text = tk.Text(self.weighted_tab, height=4, wrap=tk.WORD,
                                  state='disabled', padx=5, pady=5)
        
        # Buttons
        self.button_frame = ttk.Frame(self.root)
        self.analyze_button = ttk.Button(self.button_frame, text="Start Analysis", 
                                       command=self.start_analysis, width=12)
        self.knn_button = ttk.Button(self.button_frame, text="KNN Analysis", 
                                   command=self.run_knn_analysis, state=tk.DISABLED, width=12)
        self.weighted_button = ttk.Button(self.button_frame, text="Weighted Vote", 
                                       command=self.run_weighted_vote, state=tk.DISABLED, width=12)
        self.reset_button = ttk.Button(self.button_frame, text="Reset", 
                                     command=self.reset_analysis, width=12)

    def make_progress_section(self, label_text, status_text):
        frame = ttk.Frame(self.progress_frame, padding=(0, 2, 0, 2))
        label = ttk.Label(frame, text=label_text)
        progress = ttk.Progressbar(frame, mode='determinate', length=250)
        status = ttk.Label(frame, text=status_text, foreground="gray")
        frame.pack(fill=tk.X, padx=2, pady=1)
        label.pack(anchor=tk.W)
        progress.pack(fill=tk.X, pady=1)
        status.pack(anchor=tk.W)
        frame.label = label
        frame.progress = progress
        frame.status = status
        return frame

    def setup_layout(self):
        self.header.pack(pady=5)
        self.url_frame.pack(pady=2, padx=5, fill=tk.X)
        self.url_label.pack(side=tk.LEFT)
        self.url_entry.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        self.progress_frame.pack(pady=5, padx=5, fill=tk.X)
        
        # Notebook setup
        self.notebook.add(self.results_tab, text="Analysis Results")
        self.notebook.add(self.knn_tab, text="KNN Prediction")
        self.notebook.add(self.weighted_tab, text="Weighted Vote")
        self.notebook.pack(pady=5, padx=5, fill=tk.BOTH, expand=True)
        
        # Results tab layout
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.result_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # KNN tab layout
        self.knn_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Weighted tab layout
        self.weighted_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Buttons layout
        self.button_frame.pack(pady=5)
        self.analyze_button.pack(side=tk.LEFT, padx=2)
        self.knn_button.pack(side=tk.LEFT, padx=2)
        self.weighted_button.pack(side=tk.LEFT, padx=2)
        self.reset_button.pack(side=tk.LEFT, padx=2)

    def start_analysis(self):
        youtube_url = self.url_entry.get()
        if not youtube_url:
            messagebox.showerror("Error", "Please enter a YouTube URL")
            return
            
        self.reset_analysis()
        self.analyze_button.config(state=tk.DISABLED)
        self.knn_button.config(state=tk.DISABLED)
        self.weighted_button.config(state=tk.DISABLED)
        
        try:
            # Download video
            self.update_step(self.download_frame, "Downloading...", 0)
            video_id = download_youtube_video(youtube_url)
            if not video_id:
                raise Exception("Failed to download video")
                
            for i in range(0, 101, 10):
                self.update_step(self.download_frame, f"Downloading... {i}%", i)
                self.root.update()
                time.sleep(0.1)
            self.update_step(self.download_frame, "Download complete!", 100, "green")
            
            # Video Analysis
            self.update_step(self.video_frame, "Starting analysis...", 0)
            video_path = f"download/videos/{video_id}.mp4"
            for i in range(0, 51, 10):
                self.update_step(self.video_frame, f"Processing frames... {i}%", i)
                self.root.update()
                time.sleep(0.1)
                
            video_pred, video_conf = predict_video(video_path)
            for i in range(50, 101, 10):
                self.update_step(self.video_frame, f"Finalizing... {i}%", i)
                self.root.update()
                time.sleep(0.05)
            self.update_step(self.video_frame, f"Analysis complete", 100, "green")
            
            # Audio Analysis
            self.update_step(self.audio_frame, "Starting analysis...", 0)
            audio_path = f"download/audio/{video_id}.mp3"
            for i in range(0, 51, 10):
                self.update_step(self.audio_frame, f"Processing audio... {i}%", i)
                self.root.update()
                time.sleep(0.1)
                
            audio_pred, audio_conf = predict_audio(audio_path)
            # Adjust audio prediction from 0-5 to 1-6
            if audio_pred is not None:
                audio_pred = int(audio_pred) + 1
                audio_pred = str(audio_pred)
            for i in range(50, 101, 10):
                self.update_step(self.audio_frame, f"Finalizing... {i}%", i)
                self.root.update()
                time.sleep(0.05)
            self.update_step(self.audio_frame, f"Analysis complete", 100, "green")

            # Comment Analysis
            self.update_step(self.comment_frame, "Starting analysis...", 0)
            comment_path = f"download/comments/{video_id}.json"
            for i in range(0, 51, 10):
                self.update_step(self.comment_frame, f"Processing comments... {i}%", i)
                self.root.update()
                time.sleep(0.1)

            comment_pred, comment_conf = predict_comments(comment_path)
            for i in range(50, 101, 10):
                self.update_step(self.comment_frame, f"Finalizing... {i}%", i)
                self.root.update()
                time.sleep(0.05)
            self.update_step(self.comment_frame, f"Analysis complete", 100, "green")

            # Store results
            self.current_results = {
                'video_pred': video_pred,
                'video_conf': video_conf,
                'audio_pred': audio_pred,
                'audio_conf': audio_conf,
                'comment_pred': comment_pred,
                'comment_conf': comment_conf
            }

            # Display results
            self.show_results(video_pred, video_conf, audio_pred, audio_conf, 
                            comment_pred, comment_conf)
            
            # Enable analysis buttons if we have at least one valid prediction
            if any([video_pred, audio_pred, comment_pred]):
                self.weighted_button.config(state=tk.NORMAL)
                if all([video_pred, audio_pred, comment_pred]) and self.knn_model:
                    self.knn_button.config(state=tk.NORMAL)
            
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
            self.reset_analysis()
        finally:
            self.analyze_button.config(state=tk.NORMAL)

    # def run_knn_analysis(self):
    #     if not hasattr(self, 'current_results'):
    #         messagebox.showerror("Error", "Please run analysis first")
    #         return
            
    #     if self.knn_model is None:
    #         messagebox.showerror("Error", "KNN model not loaded")
    #         return
            
    #     try:
    #         # Prepare features for KNN
    #         features = pd.DataFrame({
    #             'video_prediction': [self.current_results['video_pred']],
    #             'video_confidence': [self.current_results['video_conf']],
    #             'audio_prediction': [self.current_results['audio_pred']],
    #             'audio_confidence': [self.current_results['audio_conf']],
    #             'text_prediction': [self.current_results['comment_pred']],
    #             'text_confidence': [self.current_results['comment_conf']]
    #         })
            
    #         # Convert all features to numeric
    #         features = features.apply(pd.to_numeric, errors='coerce').fillna(0)
            
    #         # Make prediction
    #         knn_prediction = self.knn_model['model'].predict(features)
    #         predicted_category = self.knn_model['label_encoder'].inverse_transform(knn_prediction)[0]
            
    #         # Get category name
    #         predicted_label = self.get_category_label(predicted_category)
            
    #         # Display KNN results
    #         self.knn_text.config(state='normal')
    #         self.knn_text.delete(1.0, tk.END)
            
    #         knn_results = "=== KNN FINAL PREDICTION ===\n\n"
    #         knn_results += f"Based on combined analysis of all modalities:\n"
    #         knn_results += f"  - Predicted Content: {predicted_label}\n"
    #         knn_results += f"  - Predicted Category: {predicted_category}\n\n"
    #         knn_results += "Note: KNN prediction requires all three modalities"
            
    #         self.knn_text.insert(tk.END, knn_results)
    #         self.knn_text.config(state='disabled')
            
    #         # Switch to KNN tab
    #         self.notebook.select(self.knn_tab)
            
    #     except Exception as e:
    #         messagebox.showerror("Error", f"KNN analysis failed: {str(e)}")


    def run_knn_analysis(self):
        if not hasattr(self, 'current_results'):
            messagebox.showerror("Error", "Please run analysis first")
            return
            
        if self.knn_model is None:
            messagebox.showerror("Error", "KNN model not loaded")
            return
            
        try:
            # Prepare features for KNN
            features = pd.DataFrame({
                'video_prediction': [self.current_results['video_pred']],
                'video_confidence': [self.current_results['video_conf']],
                'audio_prediction': [self.current_results['audio_pred']],
                'audio_confidence': [self.current_results['audio_conf']],
                'text_prediction': [self.current_results['comment_pred']],
                'text_confidence': [self.current_results['comment_conf']]
            })
            
            # Convert all features to numeric
            features = features.apply(pd.to_numeric, errors='coerce').fillna(0)
            
            # Make prediction and get probabilities
            knn_prediction = self.knn_model['model'].predict(features)
            predicted_probs = self.knn_model['model'].predict_proba(features)[0]
            predicted_category = self.knn_model['label_encoder'].inverse_transform(knn_prediction)[0]
            
            # Get confidence (probability) of the predicted class
            confidence = predicted_probs.max() * 100  # Convert to percentage
            
            # Get category name
            predicted_label = self.get_category_label(predicted_category)
            
            # Display KNN results with confidence
            self.knn_text.config(state='normal')
            self.knn_text.delete(1.0, tk.END)
            
            knn_results = "=== KNN FINAL PREDICTION ===\n\n"
            knn_results += f"Based on combined analysis of all modalities:\n"
            knn_results += f"  - Predicted Content: {predicted_label}\n"
            knn_results += f"  - Predicted Category: {predicted_category}\n"
            knn_results += f"  - Confidence: {round(confidence, 2)}%\n\n"
            knn_results += "Note: KNN prediction requires all three modalities"
            
            self.knn_text.insert(tk.END, knn_results)
            self.knn_text.config(state='disabled')
            
            # Switch to KNN tab
            self.notebook.select(self.knn_tab)
            
        except Exception as e:
            messagebox.showerror("Error", f"KNN analysis failed: {str(e)}")

    def run_weighted_vote(self):
        if not hasattr(self, 'current_results'):
            messagebox.showerror("Error", "Please run analysis first")
            return
            
        try:
            # Initialize variables
            video_pred = video_conf = None
            audio_pred = audio_conf = None
            comment_pred = comment_conf = None
            
            # Get available predictions
            if self.current_results['video_pred'] is not None:
                video_pred = int(self.current_results['video_pred'])
                video_conf = self.current_results['video_conf'] / 100
                
            if self.current_results['audio_pred'] is not None:
                audio_pred = int(self.current_results['audio_pred'])
                audio_conf = self.current_results['audio_conf'] / 100
                
            if self.current_results['comment_pred'] is not None:
                comment_pred = int(self.current_results['comment_pred'])
                comment_conf = self.current_results['comment_conf'] / 100
            
            # Determine which modalities are available
            available_modalities = []
            if video_pred is not None:
                available_modalities.append('video')
            if audio_pred is not None:
                available_modalities.append('audio')
            if comment_pred is not None:
                available_modalities.append('comment')
                
            if not available_modalities:
                messagebox.showerror("Error", "No valid predictions available for weighted vote")
                return
                
            # Define base weights
            base_weights = {
                'video': 0.5,
                'audio': 0.3,
                'comment': 0.2
            }
            
            # Adjust weights based on available modalities
            total_weight = sum(base_weights[mod] for mod in available_modalities)
            adjusted_weights = {mod: base_weights[mod]/total_weight for mod in available_modalities}
            
            # Create weighted votes
            weighted_votes = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
            
            # Add votes from available modalities
            if 'video' in adjusted_weights:
                weighted_votes[video_pred] += adjusted_weights['video'] * video_conf
                
            if 'audio' in adjusted_weights:
                weighted_votes[audio_pred] += adjusted_weights['audio'] * audio_conf
                
            if 'comment' in adjusted_weights:
                weighted_votes[comment_pred] += adjusted_weights['comment'] * comment_conf
            
            # Get the prediction with highest weighted vote
            final_prediction = max(weighted_votes.items(), key=lambda x: x[1])[0]
            final_confidence = weighted_votes[final_prediction] * 100
            
            # Get category name
            predicted_label = self.get_category_label(final_prediction)
            
            # Display weighted vote results
            self.weighted_text.config(state='normal')
            self.weighted_text.delete(1.0, tk.END)
            
            weighted_results = "=== WEIGHTED VOTE PREDICTION ===\n\n"
            weighted_results += f"Based on weighted combination of available modalities:\n"
            weighted_results += f"  - Predicted Content: {predicted_label}\n"
            weighted_results += f"  - Predicted Category: {final_prediction}\n"
            weighted_results += f"  - Combined Confidence: {round(final_confidence, 2)}%\n\n"
            weighted_results += "Weights used:\n"
            
            for mod in available_modalities:
                weighted_results += f"  - {mod.capitalize()}: {adjusted_weights[mod]*100:.1f}%\n"
                
            if len(available_modalities) < 3:
                weighted_results += f"\nNote: Used {len(available_modalities)} out of 3 modalities"
            
            self.weighted_text.insert(tk.END, weighted_results)
            self.weighted_text.config(state='disabled')
            
            # Switch to weighted vote tab
            self.notebook.select(self.weighted_tab)
            
        except Exception as e:
            messagebox.showerror("Error", f"Weighted vote analysis failed: {str(e)}")

    def get_category_label(self, category):
        label_map = {
            1: 'Baseball Bat Attack',
            2: 'Bomb Explosion in Public',
            3: 'Hit and Run',
            4: 'Kill Animals',
            5: 'Heavy Lip Kissing',
            6: 'None'
        }
        try:
            return label_map.get(int(category), 'Unknown')
        except (ValueError, TypeError):
            return 'Unknown'

    def update_step(self, frame, text, value, color="black"):
        frame.progress['value'] = value
        frame.status.config(text=text, foreground=color)
        self.root.update()

    def show_results(self, video_pred, video_conf, audio_pred, audio_conf, comment_pred, comment_conf):
        video_label = self.get_category_label(video_pred)
        audio_label = self.get_category_label(audio_pred)
        comment_label = self.get_category_label(comment_pred)

        self.result_text.config(state='normal')
        self.result_text.delete(1.0, tk.END)
        
        results = "=== ANALYSIS RESULTS ===\n\n"
        results += f"Video Analysis:\n"
        results += f"  - Content: {video_label}\n"
        results += f"  - Confidence: {round(video_conf, 2)}%\n\n"
        results += f"Audio Analysis:\n"
        results += f"  - Content: {audio_label}\n"
        results += f"  - Confidence: {round(audio_conf, 2)}%\n\n"
        results += f"Comment Analysis:\n"
        results += f"  - Content: {comment_label}\n"
        results += f"  - Confidence: {round(comment_conf, 2)}%\n\n"
        
        # Show which predictions are available
        available = []
        if video_pred: available.append("Video")
        if audio_pred: available.append("Audio")
        if comment_pred: available.append("Comments")
        
        results += f"Available predictions: {', '.join(available) if available else 'None'}\n"
        results += "Click 'Weighted Vote' for final prediction"
        
        if all([video_pred, audio_pred, comment_pred]):
            results += " or 'KNN Analysis' for combined prediction"
        
        self.result_text.insert(tk.END, results)
        self.result_text.config(state='disabled')

    def reset_analysis(self):
        for frame in [self.download_frame, self.video_frame, self.audio_frame, self.comment_frame]:
            frame.progress['value'] = 0
            frame.status.config(text="Waiting", foreground="gray")
        
        self.result_text.config(state='normal')
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "Results will appear here after analysis...")
        self.result_text.config(state='disabled')
        
        self.knn_text.config(state='normal')
        self.knn_text.delete(1.0, tk.END)
        self.knn_text.insert(tk.END, "Run analysis first, then click 'KNN Analysis'")
        self.knn_text.config(state='disabled')
        
        self.weighted_text.config(state='normal')
        self.weighted_text.delete(1.0, tk.END)
        self.weighted_text.insert(tk.END, "Run analysis first, then click 'Weighted Vote'")
        self.weighted_text.config(state='disabled')
        
        self.knn_button.config(state=tk.DISABLED)
        self.weighted_button.config(state=tk.DISABLED)
        self.notebook.select(self.results_tab)
        
        if hasattr(self, 'current_results'):
            del self.current_results
            
        self.root.update()

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoAnalyzerApp(root)
    root.mainloop()