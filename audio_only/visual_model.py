import time
import tkinter as tk
from tkinter import ttk, messagebox
from video_d import main as download_youtube_video
from video_predictor import predict_video
import os
import threading
import cv2
from PIL import Image, ImageTk

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

        # Notebook for tabs
        self.notebook = ttk.Notebook(self.left_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Analysis Results Tab
        self.results_tab = ttk.Frame(self.notebook)
        self.result_text = tk.Text(self.results_tab, wrap=tk.WORD, state='disabled', padx=5, pady=5)
        self.result_scroll = ttk.Scrollbar(self.results_tab, command=self.result_text.yview)
        self.result_text.config(yscrollcommand=self.result_scroll.set)
        
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
        
        self.reset_button = ttk.Button(self.button_frame, text="Reset", 
                                     command=self.reset_analysis, width=15)
        self.reset_button.pack(side=tk.TOP, fill=tk.X, pady=2)
        
    def setup_layout(self):
        # Notebook tabs
        self.notebook.add(self.results_tab, text="Analysis Results")
        
        # Results tab layout
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.result_scroll.pack(side=tk.RIGHT, fill=tk.Y)

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
            
            # Store results
            self.current_results = {
                'video_pred': video_pred,
                'video_conf': video_conf
            }

            # Display results
            self.show_results(video_pred, video_conf)
            
            # Update summary
            self.update_summary(video_pred)
            
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
            self.reset_analysis()
        finally:
            self.analyze_button.config(state=tk.NORMAL)

    def update_summary(self, video_pred):
        summary = "Analysis Summary:\n\n"
        
        if video_pred:
            summary += f"Video: {self.get_category_label(video_pred)}\n"
        else:
            summary += "Video: No prediction\n"
            
        self.summary_label.config(text=summary)

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

    def show_results(self, video_pred, video_conf):
        video_label = self.get_category_label(video_pred)

        self.result_text.config(state='normal')
        self.result_text.delete(1.0, tk.END)
        
        results = "=== ANALYSIS RESULTS ===\n\n"
        results += f"Video Analysis:\n"
        results += f"  - Content: {video_label}\n"
        results += f"  - Confidence: {round(video_conf, 2)}%\n\n"
        
        self.result_text.insert(tk.END, results)
        self.result_text.config(state='disabled')

    def reset_analysis(self):
        for frame in [self.download_frame, self.video_frame]:
            frame.progress['value'] = 0
            frame.status.config(text="Waiting", foreground="gray")
        
        self.result_text.config(state='normal')
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "Results will appear here after analysis...")
        self.result_text.config(state='disabled')
        
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