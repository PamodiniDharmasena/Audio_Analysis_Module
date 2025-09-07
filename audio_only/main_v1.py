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
# from PIL import Image, ImageTk
# import cv2
# import os
# import threading

# class VideoPlayer:
#     def __init__(self, parent):
#         self.parent = parent
#         self.video_path = None
#         self.cap = None
#         self.playing = False
#         self.current_frame = 0
#         self.total_frames = 0
#         self.delay = 15  # ms delay between frames
#         self.video_width = 320
#         self.video_height = 180
        
#         # Create video player frame
#         self.frame = ttk.LabelFrame(parent, text="Video Preview", padding=5)
#         self.frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
#         # Video display
#         self.video_label = ttk.Label(self.frame)
#         self.video_label.pack(pady=5)
        
#         # Controls frame
#         self.controls_frame = ttk.Frame(self.frame)
#         self.controls_frame.pack(fill=tk.X, pady=5)
        
#         # Buttons
#         self.play_button = ttk.Button(self.controls_frame, text="▶", width=3, command=self.play_video)
#         self.play_button.pack(side=tk.LEFT, padx=2)
        
#         self.pause_button = ttk.Button(self.controls_frame, text="⏸", width=3, command=self.pause_video)
#         self.pause_button.pack(side=tk.LEFT, padx=2)
        
#         self.stop_button = ttk.Button(self.controls_frame, text="⏹", width=3, command=self.stop_video)
#         self.stop_button.pack(side=tk.LEFT, padx=2)
        
#         # Progress slider
#         self.progress_slider = ttk.Scale(self.controls_frame, from_=0, to=100, 
#                                        orient=tk.HORIZONTAL, command=self.on_slider_change)
#         self.progress_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
#         # Time label
#         self.time_label = ttk.Label(self.controls_frame, text="00:00", font=('Arial', 8))
#         self.time_label.pack(side=tk.LEFT, padx=2)
    
#     def load_video(self, video_path):
#         self.video_path = video_path
#         if self.cap is not None:
#             self.cap.release()
            
#         self.cap = cv2.VideoCapture(video_path)
#         if not self.cap.isOpened():
#             messagebox.showerror("Error", "Could not open video file")
#             return False
            
#         self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         self.current_frame = 0
#         self.progress_slider.config(to=self.total_frames)
#         self.update_time_label()
#         self.show_frame()
#         return True
    
#     def play_video(self):
#         if self.cap is None:
#             return
            
#         self.playing = True
#         self.play_thread()
    
#     def play_thread(self):
#         def _play():
#             while self.playing and self.current_frame < self.total_frames:
#                 ret, frame = self.cap.read()
#                 if not ret:
#                     break
                    
#                 self.current_frame += 1
#                 self.progress_slider.set(self.current_frame)
#                 self.update_time_label()
                
#                 # Convert frame to RGB and display
#                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 frame = cv2.resize(frame, (self.video_width, self.video_height))
#                 img = Image.fromarray(frame)
#                 imgtk = ImageTk.PhotoImage(image=img)
                
#                 self.video_label.imgtk = imgtk
#                 self.video_label.config(image=imgtk)
                
#                 time.sleep(self.delay / 1000)
            
#             if self.current_frame >= self.total_frames:
#                 self.stop_video()
        
#         if self.playing:
#             thread = threading.Thread(target=_play, daemon=True)
#             thread.start()
    
#     def pause_video(self):
#         self.playing = False
    
#     def stop_video(self):
#         self.playing = False
#         if self.cap is not None:
#             self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#             self.current_frame = 0
#             self.progress_slider.set(0)
#             self.update_time_label()
#             self.show_frame()
    
#     def show_frame(self):
#         if self.cap is None:
#             return
            
#         self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
#         ret, frame = self.cap.read()
#         if ret:
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frame = cv2.resize(frame, (self.video_width, self.video_height))
#             img = Image.fromarray(frame)
#             imgtk = ImageTk.PhotoImage(image=img)
            
#             self.video_label.imgtk = imgtk
#             self.video_label.config(image=imgtk)
    
#     def on_slider_change(self, value):
#         if self.cap is None:
#             return
            
#         try:
#             frame_num = int(float(value))
#             if frame_num != self.current_frame:
#                 self.current_frame = frame_num
#                 self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
#                 self.update_time_label()
#                 self.show_frame()
#         except ValueError:
#             pass
    
#     def update_time_label(self):
#         if self.cap is None:
#             return
            
#         fps = self.cap.get(cv2.CAP_PROP_FPS)
#         if fps == 0:
#             fps = 30  # default if fps not available
            
#         current_time = self.current_frame / fps
#         total_time = self.total_frames / fps
        
#         current_min, current_sec = divmod(int(current_time), 60)
        
#         self.time_label.config(text=f"{current_min:02d}:{current_sec:02d}")
    
#     def release(self):
#         self.playing = False
#         if self.cap is not None:
#             self.cap.release()

# class VideoAnalyzerApp:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("Video Content Analyzer")
#         self.root.geometry("1000x800")
        
#         # Load KNN model
#         try:
#             self.knn_model = joblib.load("knn_model.joblib")
#         except Exception as e:
#             messagebox.showerror("Error", f"Failed to load KNN model: {str(e)}")
#             self.knn_model = None
        
#         self.create_widgets()
#         self.setup_layout()
        
#     def create_widgets(self):
#         # Main container frame
#         self.main_frame = ttk.Frame(self.root)
#         self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
#         # Left column - Analysis
#         self.left_frame = ttk.Frame(self.main_frame)
#         self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
#         # Right column - Video
#         self.right_frame = ttk.Frame(self.main_frame, width=350)
#         self.right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5)
        
#         # Left column widgets
#         self.create_left_column()
        
#         # Right column widgets
#         self.create_right_column()
        
#     def create_left_column(self):
#         # URL Entry
#         self.url_frame = ttk.Frame(self.left_frame)
#         self.url_frame.pack(fill=tk.X, pady=5)
        
#         self.url_label = ttk.Label(self.url_frame, text="YouTube URL:")
#         self.url_label.pack(side=tk.LEFT)
        
#         self.url_entry = ttk.Entry(self.url_frame)
#         self.url_entry.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        
#         # Progress Tracking
#         self.progress_frame = ttk.LabelFrame(self.left_frame, text="Processing Steps", padding=5)
#         self.progress_frame.pack(fill=tk.X, pady=5)
        
#         self.download_frame = self.make_progress_section("1. Downloading video...", "Waiting")
#         self.video_frame = self.make_progress_section("2. Analyzing video content...", "Waiting")
#         self.audio_frame = self.make_progress_section("3. Analyzing audio content...", "Waiting")
#         self.comment_frame = self.make_progress_section("4. Analyzing comment content...", "Waiting")

#         # Notebook for tabs
#         self.notebook = ttk.Notebook(self.left_frame)
#         self.notebook.pack(fill=tk.BOTH, expand=True, pady=5)
        
#         # Analysis Results Tab
#         self.results_tab = ttk.Frame(self.notebook)
#         self.result_text = tk.Text(self.results_tab, wrap=tk.WORD, state='disabled', padx=5, pady=5)
#         self.result_scroll = ttk.Scrollbar(self.results_tab, command=self.result_text.yview)
#         self.result_text.config(yscrollcommand=self.result_scroll.set)
        
#         # KNN Results Tab
#         self.knn_tab = ttk.Frame(self.notebook)
#         self.knn_text = tk.Text(self.knn_tab, wrap=tk.WORD, state='disabled', padx=5, pady=5)
#         self.knn_scroll = ttk.Scrollbar(self.knn_tab, command=self.knn_text.yview)
#         self.knn_text.config(yscrollcommand=self.knn_scroll.set)
        
#         # Weighted Vote Results Tab
#         self.weighted_tab = ttk.Frame(self.notebook)
#         self.weighted_text = tk.Text(self.weighted_tab, wrap=tk.WORD, state='disabled', padx=5, pady=5)
#         self.weighted_scroll = ttk.Scrollbar(self.weighted_tab, command=self.weighted_text.yview)
#         self.weighted_text.config(yscrollcommand=self.weighted_scroll.set)
        
#         # Buttons
#         self.button_frame = ttk.Frame(self.left_frame)
#         self.button_frame.pack(fill=tk.X, pady=5)
        
#         self.analyze_button = ttk.Button(self.button_frame, text="Start Analysis", 
#                                       command=self.start_analysis, width=12)
#         self.analyze_button.pack(side=tk.LEFT, padx=2)
        
#         self.knn_button = ttk.Button(self.button_frame, text="KNN Analysis", 
#                                    command=self.run_knn_analysis, state=tk.DISABLED, width=12)
#         self.knn_button.pack(side=tk.LEFT, padx=2)
        
#         self.weighted_button = ttk.Button(self.button_frame, text="Weighted Vote", 
#                                        command=self.run_weighted_vote, state=tk.DISABLED, width=12)
#         self.weighted_button.pack(side=tk.LEFT, padx=2)
        
#         self.reset_button = ttk.Button(self.button_frame, text="Reset", 
#                                      command=self.reset_analysis, width=12)
#         self.reset_button.pack(side=tk.LEFT, padx=2)
        
#     def create_right_column(self):
#         # Video Player
#         self.video_player = VideoPlayer(self.right_frame)
        
#         # Analysis Summary Frame
#         self.summary_frame = ttk.LabelFrame(self.right_frame, text="Analysis Summary", padding=10)
#         self.summary_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
#         self.summary_label = ttk.Label(self.summary_frame, text="No analysis performed yet")
#         self.summary_label.pack(anchor=tk.NW)
        
#     def setup_layout(self):
#         # Notebook tabs
#         self.notebook.add(self.results_tab, text="Analysis Results")
#         self.notebook.add(self.knn_tab, text="KNN Prediction")
#         self.notebook.add(self.weighted_tab, text="Weighted Vote")
        
#         # Results tab layout
#         self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
#         self.result_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
#         # KNN tab layout
#         self.knn_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
#         self.knn_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
#         # Weighted tab layout
#         self.weighted_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
#         self.weighted_scroll.pack(side=tk.RIGHT, fill=tk.Y)

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
            
#             # Load video into player
#             video_path = f"download/videos/{video_id}.mp4"
#             if os.path.exists(video_path):
#                 self.video_player.load_video(video_path)
            
#             # Video Analysis
#             self.update_step(self.video_frame, "Starting analysis...", 0)
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
#             # Adjust audio prediction from 0-5 to 1-6
#             if audio_pred is not None:
#                 audio_pred = int(audio_pred) + 1
#                 audio_pred = str(audio_pred)
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

#             # Store results
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
            
#             # Update summary
#             self.update_summary(video_pred, audio_pred, comment_pred)
            
#             # Enable analysis buttons if we have at least one valid prediction
#             if any([video_pred, audio_pred, comment_pred]):
#                 self.weighted_button.config(state=tk.NORMAL)
#                 if all([video_pred, audio_pred, comment_pred]) and self.knn_model:
#                     self.knn_button.config(state=tk.NORMAL)
            
#         except Exception as e:
#             messagebox.showerror("Error", f"Analysis failed: {str(e)}")
#             self.reset_analysis()
#         finally:
#             self.analyze_button.config(state=tk.NORMAL)

#     def update_summary(self, video_pred, audio_pred, comment_pred):
#         summary = "Analysis Summary:\n\n"
        
#         if video_pred:
#             summary += f"Video: {self.get_category_label(video_pred)}\n"
#         else:
#             summary += "Video: No prediction\n"
            
#         if audio_pred:
#             summary += f"Audio: {self.get_category_label(audio_pred)}\n"
#         else:
#             summary += "Audio: No prediction\n"
            
#         if comment_pred:
#             summary += f"Comments: {self.get_category_label(comment_pred)}\n"
#         else:
#             summary += "Comments: No prediction\n"
            
#         self.summary_label.config(text=summary)

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
            
#             # Make prediction and get probabilities
#             knn_prediction = self.knn_model['model'].predict(features)
#             predicted_probs = self.knn_model['model'].predict_proba(features)[0]
#             predicted_category = self.knn_model['label_encoder'].inverse_transform(knn_prediction)[0]
            
#             # Get confidence (probability) of the predicted class
#             confidence = predicted_probs.max() * 100  # Convert to percentage
            
#             # Get category name
#             predicted_label = self.get_category_label(predicted_category)
            
#             # Display KNN results with confidence
#             self.knn_text.config(state='normal')
#             self.knn_text.delete(1.0, tk.END)
            
#             knn_results = "=== KNN FINAL PREDICTION ===\n\n"
#             knn_results += f"Based on combined analysis of all modalities:\n"
#             knn_results += f"  - Predicted Content: {predicted_label}\n"
#             knn_results += f"  - Predicted Category: {predicted_category}\n"
#             knn_results += f"  - Confidence: {round(confidence, 2)}%\n\n"
#             knn_results += "Note: KNN prediction requires all three modalities"
            
#             self.knn_text.insert(tk.END, knn_results)
#             self.knn_text.config(state='disabled')
            
#             # Update summary
#             self.summary_label.config(text=f"Final KNN Prediction:\n{predicted_label}\n({predicted_category})")
            
#             # Switch to KNN tab
#             self.notebook.select(self.knn_tab)
            
#         except Exception as e:
#             messagebox.showerror("Error", f"KNN analysis failed: {str(e)}")

#     def run_weighted_vote(self):
#         if not hasattr(self, 'current_results'):
#             messagebox.showerror("Error", "Please run analysis first")
#             return
            
#         try:
#             # Initialize variables
#             video_pred = video_conf = None
#             audio_pred = audio_conf = None
#             comment_pred = comment_conf = None
            
#             # Get available predictions
#             if self.current_results['video_pred'] is not None:
#                 video_pred = int(self.current_results['video_pred'])
#                 video_conf = self.current_results['video_conf'] / 100
                
#             if self.current_results['audio_pred'] is not None:
#                 audio_pred = int(self.current_results['audio_pred'])
#                 audio_conf = self.current_results['audio_conf'] / 100
                
#             if self.current_results['comment_pred'] is not None:
#                 comment_pred = int(self.current_results['comment_pred'])
#                 comment_conf = self.current_results['comment_conf'] / 100
            
#             # Determine which modalities are available
#             available_modalities = []
#             if video_pred is not None:
#                 available_modalities.append('video')
#             if audio_pred is not None:
#                 available_modalities.append('audio')
#             if comment_pred is not None:
#                 available_modalities.append('comment')
                
#             if not available_modalities:
#                 messagebox.showerror("Error", "No valid predictions available for weighted vote")
#                 return
                
#             # Define base weights
#             base_weights = {
#                 'video': 0.5,
#                 'audio': 0.3,
#                 'comment': 0.2
#             }
            
#             # Adjust weights based on available modalities
#             total_weight = sum(base_weights[mod] for mod in available_modalities)
#             adjusted_weights = {mod: base_weights[mod]/total_weight for mod in available_modalities}
            
#             # Create weighted votes
#             weighted_votes = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
            
#             # Add votes from available modalities
#             if 'video' in adjusted_weights:
#                 weighted_votes[video_pred] += adjusted_weights['video'] * video_conf
                
#             if 'audio' in adjusted_weights:
#                 weighted_votes[audio_pred] += adjusted_weights['audio'] * audio_conf
                
#             if 'comment' in adjusted_weights:
#                 weighted_votes[comment_pred] += adjusted_weights['comment'] * comment_conf
            
#             # Get the prediction with highest weighted vote
#             final_prediction = max(weighted_votes.items(), key=lambda x: x[1])[0]
#             final_confidence = weighted_votes[final_prediction] * 100
            
#             # Get category name
#             predicted_label = self.get_category_label(final_prediction)
            
#             # Display weighted vote results
#             self.weighted_text.config(state='normal')
#             self.weighted_text.delete(1.0, tk.END)
            
#             weighted_results = "=== WEIGHTED VOTE PREDICTION ===\n\n"
#             weighted_results += f"Based on weighted combination of available modalities:\n"
#             weighted_results += f"  - Predicted Content: {predicted_label}\n"
#             weighted_results += f"  - Predicted Category: {final_prediction}\n"
#             weighted_results += f"  - Combined Confidence: {round(final_confidence, 2)}%\n\n"
#             weighted_results += "Weights used:\n"
            
#             for mod in available_modalities:
#                 weighted_results += f"  - {mod.capitalize()}: {adjusted_weights[mod]*100:.1f}%\n"
                
#             if len(available_modalities) < 3:
#                 weighted_results += f"\nNote: Used {len(available_modalities)} out of 3 modalities"
            
#             self.weighted_text.insert(tk.END, weighted_results)
#             self.weighted_text.config(state='disabled')
            
#             # Update summary
#             self.summary_label.config(text=f"Final Weighted Prediction:\n{predicted_label}\n({final_prediction})")
            
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
#         try:
#             return label_map.get(int(category), 'Unknown')
#         except (ValueError, TypeError):
#             return 'Unknown'

#     def update_step(self, frame, text, value, color="black"):
#         frame.progress['value'] = value
#         frame.status.config(text=text, foreground=color)
#         self.root.update()

#     def show_results(self, video_pred, video_conf, audio_pred, audio_conf, comment_pred, comment_conf):
#         video_label = self.get_category_label(video_pred)
#         audio_label = self.get_category_label(audio_pred)
#         comment_label = self.get_category_label(comment_pred)

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
        
#         # Show which predictions are available
#         available = []
#         if video_pred: available.append("Video")
#         if audio_pred: available.append("Audio")
#         if comment_pred: available.append("Comments")
        
#         results += f"Available predictions: {', '.join(available) if available else 'None'}\n"
#         results += "Click 'Weighted Vote' for final prediction"
        
#         if all([video_pred, audio_pred, comment_pred]):
#             results += " or 'KNN Analysis' for combined prediction"
        
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
            
#         # Reset video player
#         self.video_player.stop_video()
#         self.video_player.video_label.config(image=None)
#         self.video_player.time_label.config(text="00:00")
#         self.video_player.progress_slider.set(0)
        
#         # Reset summary
#         self.summary_label.config(text="No analysis performed yet")
            
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
from PIL import Image, ImageTk
import cv2
import os
import threading

class VideoPlayer:
    def __init__(self, parent):
        self.parent = parent
        self.video_path = None
        self.cap = None
        self.playing = False
        self.current_frame = 0
        self.total_frames = 0
        self.delay = 15  # ms delay between frames
        self.video_width = 320
        self.video_height = 180
        
        # Create video player frame
        self.frame = ttk.LabelFrame(parent, text="Video Preview", padding=5)
        self.frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Video display
        self.video_label = ttk.Label(self.frame)
        self.video_label.pack(pady=5)
        
        # Controls frame
        self.controls_frame = ttk.Frame(self.frame)
        self.controls_frame.pack(fill=tk.X, pady=5)
        
        # Buttons
        self.play_button = ttk.Button(self.controls_frame, text="▶", width=3, command=self.play_video)
        self.play_button.pack(side=tk.LEFT, padx=2)
        
        self.pause_button = ttk.Button(self.controls_frame, text="⏸", width=3, command=self.pause_video)
        self.pause_button.pack(side=tk.LEFT, padx=2)
        
        self.stop_button = ttk.Button(self.controls_frame, text="⏹", width=3, command=self.stop_video)
        self.stop_button.pack(side=tk.LEFT, padx=2)
        
        # Progress slider
        self.progress_slider = ttk.Scale(self.controls_frame, from_=0, to=100, 
                                       orient=tk.HORIZONTAL, command=self.on_slider_change)
        self.progress_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Time label
        self.time_label = ttk.Label(self.controls_frame, text="00:00", font=('Arial', 8))
        self.time_label.pack(side=tk.LEFT, padx=2)
    
    def load_video(self, video_path):
        self.video_path = video_path
        if self.cap is not None:
            self.cap.release()
            
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open video file")
            return False
            
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0
        self.progress_slider.config(to=self.total_frames)
        self.update_time_label()
        self.show_frame()
        return True
    
    def play_video(self):
        if self.cap is None:
            return
            
        self.playing = True
        self.play_thread()
    
    def play_thread(self):
        def _play():
            while self.playing and self.current_frame < self.total_frames:
                ret, frame = self.cap.read()
                if not ret:
                    break
                    
                self.current_frame += 1
                self.progress_slider.set(self.current_frame)
                self.update_time_label()
                
                # Convert frame to RGB and display
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.video_width, self.video_height))
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                
                self.video_label.imgtk = imgtk
                self.video_label.config(image=imgtk)
                
                time.sleep(self.delay / 1000)
            
            if self.current_frame >= self.total_frames:
                self.stop_video()
        
        if self.playing:
            thread = threading.Thread(target=_play, daemon=True)
            thread.start()
    
    def pause_video(self):
        self.playing = False
    
    def stop_video(self):
        self.playing = False
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.current_frame = 0
            self.progress_slider.set(0)
            self.update_time_label()
            self.show_frame()
    
    def show_frame(self):
        if self.cap is None:
            return
            
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.video_width, self.video_height))
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)
    
    def on_slider_change(self, value):
        if self.cap is None:
            return
            
        try:
            frame_num = int(float(value))
            if frame_num != self.current_frame:
                self.current_frame = frame_num
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                self.update_time_label()
                self.show_frame()
        except ValueError:
            pass
    
    def update_time_label(self):
        if self.cap is None:
            return
            
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30  # default if fps not available
            
        current_time = self.current_frame / fps
        total_time = self.total_frames / fps
        
        current_min, current_sec = divmod(int(current_time), 60)
        
        self.time_label.config(text=f"{current_min:02d}:{current_sec:02d}")
    
    def release(self):
        self.playing = False
        if self.cap is not None:
            self.cap.release()

class VideoAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Content Analyzer")
        self.root.geometry("1000x800")
        
        # Load KNN model
        try:
            self.knn_model = joblib.load("knn_model.joblib")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load KNN model: {str(e)}")
            self.knn_model = None
        
        self.create_widgets()
        self.setup_layout()
        
    def create_widgets(self):
        # Main container frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left column - Analysis
        self.left_frame = ttk.Frame(self.main_frame)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Right column - Video
        self.right_frame = ttk.Frame(self.main_frame, width=350)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5)
        
        # Left column widgets
        self.create_left_column()
        
        # Right column widgets
        self.create_right_column()
        
    def create_left_column(self):
        # URL Entry
        self.url_frame = ttk.Frame(self.left_frame)
        self.url_frame.pack(fill=tk.X, pady=5)
        
        self.url_label = ttk.Label(self.url_frame, text="YouTube URL:")
        self.url_label.pack(side=tk.LEFT)
        
        self.url_entry = ttk.Entry(self.url_frame)
        self.url_entry.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        
        # Progress Tracking
        self.progress_frame = ttk.LabelFrame(self.left_frame, text="Processing Steps", padding=5)
        self.progress_frame.pack(fill=tk.X, pady=5)
        
        self.download_frame = self.make_progress_section("1. Downloading video...", "Waiting")
        self.video_frame = self.make_progress_section("2. Analyzing video content...", "Waiting")
        self.audio_frame = self.make_progress_section("3. Analyzing audio content...", "Waiting")
        self.comment_frame = self.make_progress_section("4. Analyzing comment content...", "Waiting")

        # Notebook for tabs
        self.notebook = ttk.Notebook(self.left_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Analysis Results Tab
        self.results_tab = ttk.Frame(self.notebook)
        self.result_text = tk.Text(self.results_tab, wrap=tk.WORD, state='disabled', padx=5, pady=5)
        self.result_scroll = ttk.Scrollbar(self.results_tab, command=self.result_text.yview)
        self.result_text.config(yscrollcommand=self.result_scroll.set)
        
        # KNN Results Tab
        self.knn_tab = ttk.Frame(self.notebook)
        self.knn_text = tk.Text(self.knn_tab, wrap=tk.WORD, state='disabled', padx=5, pady=5)
        self.knn_scroll = ttk.Scrollbar(self.knn_tab, command=self.knn_text.yview)
        self.knn_text.config(yscrollcommand=self.knn_scroll.set)
        
        # Weighted Vote Results Tab
        self.weighted_tab = ttk.Frame(self.notebook)
        self.weighted_text = tk.Text(self.weighted_tab, wrap=tk.WORD, state='disabled', padx=5, pady=5)
        self.weighted_scroll = ttk.Scrollbar(self.weighted_tab, command=self.weighted_text.yview)
        self.weighted_text.config(yscrollcommand=self.weighted_scroll.set)
        
    def create_right_column(self):
        # Video Player
        self.video_player = VideoPlayer(self.right_frame)
        
        # Analysis Summary Frame
        self.summary_frame = ttk.LabelFrame(self.right_frame, text="Analysis Summary", padding=10)
        self.summary_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.summary_label = ttk.Label(self.summary_frame, text="No analysis performed yet")
        self.summary_label.pack(anchor=tk.NW)
        
        # Buttons Frame in Right Column
        self.button_frame = ttk.Frame(self.right_frame)
        self.button_frame.pack(fill=tk.X, pady=5)
        
        # Buttons (now in right column)
        self.analyze_button = ttk.Button(self.button_frame, text="Start Analysis", 
                                      command=self.start_analysis, width=15)
        self.analyze_button.pack(side=tk.TOP, fill=tk.X, pady=2)
        
        self.knn_button = ttk.Button(self.button_frame, text="KNN Analysis", 
                                   command=self.run_knn_analysis, state=tk.DISABLED, width=15)
        self.knn_button.pack(side=tk.TOP, fill=tk.X, pady=2)
        
        self.weighted_button = ttk.Button(self.button_frame, text="Weighted Vote", 
                                       command=self.run_weighted_vote, state=tk.DISABLED, width=15)
        self.weighted_button.pack(side=tk.TOP, fill=tk.X, pady=2)
        
        self.reset_button = ttk.Button(self.button_frame, text="Reset", 
                                     command=self.reset_analysis, width=15)
        self.reset_button.pack(side=tk.TOP, fill=tk.X, pady=2)
        
    def setup_layout(self):
        # Notebook tabs
        self.notebook.add(self.results_tab, text="Analysis Results")
        self.notebook.add(self.knn_tab, text="KNN Prediction")
        self.notebook.add(self.weighted_tab, text="Weighted Vote")
        
        # Results tab layout
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.result_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # KNN tab layout
        self.knn_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.knn_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Weighted tab layout
        self.weighted_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.weighted_scroll.pack(side=tk.RIGHT, fill=tk.Y)

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
            
            # Load video into player
            video_path = f"download/videos/{video_id}.mp4"
            if os.path.exists(video_path):
                self.video_player.load_video(video_path)
            
            # Video Analysis
            self.update_step(self.video_frame, "Starting analysis...", 0)
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
            
            # Update summary
            self.update_summary(video_pred, audio_pred, comment_pred)
            
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

    def update_summary(self, video_pred, audio_pred, comment_pred):
        summary = "Analysis Summary:\n\n"
        
        if video_pred:
            summary += f"Video: {self.get_category_label(video_pred)}\n"
        else:
            summary += "Video: No prediction\n"
            
        if audio_pred:
            summary += f"Audio: {self.get_category_label(audio_pred)}\n"
        else:
            summary += "Audio: No prediction\n"
            
        if comment_pred:
            summary += f"Comments: {self.get_category_label(comment_pred)}\n"
        else:
            summary += "Comments: No prediction\n"
            
        self.summary_label.config(text=summary)

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
            
            # Update summary
            self.summary_label.config(text=f"Final KNN Prediction:\n{predicted_label}\n({predicted_category})")
            
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
            
            # Update summary
            self.summary_label.config(text=f"Final Weighted Prediction:\n{predicted_label}\n({final_prediction})")
            
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
            
        # Reset video player
        self.video_player.stop_video()
        self.video_player.video_label.config(image=None)
        self.video_player.time_label.config(text="00:00")
        self.video_player.progress_slider.set(0)
        
        # Reset summary
        self.summary_label.config(text="No analysis performed yet")
            
        self.root.update()

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoAnalyzerApp(root)
    root.mainloop()