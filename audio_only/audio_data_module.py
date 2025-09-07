# import time
# import tkinter as tk
# from tkinter import ttk, messagebox
# from video_d import main as download_youtube_video
# from audio_predictor import predict_audio
# import os
# import threading

# class AudioAnalyzerApp:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("Audio Content Analyzer")
#         self.root.geometry("800x600")
        
#         self.create_widgets()
#         self.setup_layout()
        
#     def create_widgets(self):
#         # Main container frame
#         self.main_frame = ttk.Frame(self.root)
#         self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
#         # URL Entry
#         self.url_frame = ttk.Frame(self.main_frame)
#         self.url_frame.pack(fill=tk.X, pady=5)
        
#         self.url_label = ttk.Label(self.url_frame, text="YouTube URL:")
#         self.url_label.pack(side=tk.LEFT)
        
#         self.url_entry = ttk.Entry(self.url_frame)
#         self.url_entry.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        
#         # Progress Tracking
#         self.progress_frame = ttk.LabelFrame(self.main_frame, text="Processing Steps", padding=5)
#         self.progress_frame.pack(fill=tk.X, pady=5)
        
#         self.download_frame = self.make_progress_section("1. Downloading ...", "Waiting")
#         self.audio_frame = self.make_progress_section("2. Extarct Audio and Analyzing audio content...", "Waiting")

#         # Results display
#         self.results_frame = ttk.LabelFrame(self.main_frame, text="Analysis Results", padding=10)
#         self.results_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
#         self.result_text = tk.Text(self.results_frame, wrap=tk.WORD, state='disabled', padx=5, pady=5)
#         self.result_scroll = ttk.Scrollbar(self.results_frame, command=self.result_text.yview)
#         self.result_text.config(yscrollcommand=self.result_scroll.set)
        
#         # Summary Frame
#         self.summary_frame = ttk.LabelFrame(self.main_frame, text="Analysis Summary", padding=10)
#         self.summary_frame.pack(fill=tk.X, pady=5)
        
#         self.summary_label = ttk.Label(self.summary_frame, text="No analysis performed yet")
#         self.summary_label.pack(anchor=tk.NW)
        
#         # Buttons Frame
#         self.button_frame = ttk.Frame(self.main_frame)
#         self.button_frame.pack(fill=tk.X, pady=5)
        
#         self.analyze_button = ttk.Button(self.button_frame, text="Start Analysis", 
#                                       command=self.start_analysis, width=15)
#         self.analyze_button.pack(side=tk.LEFT, padx=5)
        
#         self.reset_button = ttk.Button(self.button_frame, text="Reset", 
#                                      command=self.reset_analysis, width=15)
#         self.reset_button.pack(side=tk.LEFT, padx=5)
        
#     def setup_layout(self):
#         # Results frame layout
#         self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
#         self.result_scroll.pack(side=tk.RIGHT, fill=tk.Y)

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

#             # Store results
#             self.current_results = {
#                 'audio_pred': audio_pred,
#                 'audio_conf': audio_conf
#             }

#             # Display results
#             self.show_results(audio_pred, audio_conf)
            
#             # Update summary
#             self.update_summary(audio_pred)
            
#         except Exception as e:
#             messagebox.showerror("Error", f"Analysis failed: {str(e)}")
#             self.reset_analysis()
#         finally:
#             self.analyze_button.config(state=tk.NORMAL)

#     def update_summary(self, audio_pred):
#         summary = "Analysis Summary:\n\n"
        
#         if audio_pred:
#             summary += f"Audio: {self.get_category_label(audio_pred)}\n"
#         else:
#             summary += "Audio: No prediction\n"
            
#         self.summary_label.config(text=summary)

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

#     def show_results(self, audio_pred, audio_conf):
#         audio_label = self.get_category_label(audio_pred)

#         self.result_text.config(state='normal')
#         self.result_text.delete(1.0, tk.END)
        
#         results = "=== ANALYSIS RESULTS ===\n\n"
#         results += f"Audio Analysis:\n"
#         results += f"  - Content: {audio_label}\n"
#         results += f"  - Confidence: {round(audio_conf, 2)}%\n\n"
        
#         self.result_text.insert(tk.END, results)
#         self.result_text.config(state='disabled')

#     def reset_analysis(self):
#         for frame in [self.download_frame, self.audio_frame]:
#             frame.progress['value'] = 0
#             frame.status.config(text="Waiting", foreground="gray")
        
#         self.result_text.config(state='normal')
#         self.result_text.delete(1.0, tk.END)
#         self.result_text.insert(tk.END, "Results will appear here after analysis...")
#         self.result_text.config(state='disabled')
        
#         if hasattr(self, 'current_results'):
#             del self.current_results
        
#         # Reset summary
#         self.summary_label.config(text="No analysis performed yet")
            
#         self.root.update()

# if __name__ == "__main__":
#     root = tk.Tk()
#     app = AudioAnalyzerApp(root)
#     root.mainloop()

import time
import tkinter as tk
from tkinter import ttk, messagebox
from video_d import main as download_youtube_video
from audio_predictor import predict_audio
import os
import threading

class AudioAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Content Analyzer")
        self.root.geometry("800x600")
        self.root.minsize(800, 600)  # Set minimum window size
        
        # Configure grid weights for resizing
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        self.create_widgets()
        self.setup_layout()
        
        # Variable to track current URL
        self.current_url = ""
        
    def create_widgets(self):
        # Main container frame with grid layout
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Configure grid weights for main frame
        self.main_frame.grid_rowconfigure(1, weight=1)  # Results frame will expand
        self.main_frame.grid_columnconfigure(0, weight=1)
        
        # URL Entry - Row 0
        self.url_frame = ttk.Frame(self.main_frame)
        self.url_frame.grid(row=0, column=0, sticky="ew", pady=5)
        
        self.url_label = ttk.Label(self.url_frame, text="YouTube URL:")
        self.url_label.pack(side=tk.LEFT)
        
        self.url_entry = ttk.Entry(self.url_frame)
        self.url_entry.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        self.url_entry.bind("<Return>", self.on_url_entry)  # Bind Enter key
        
        # Progress Tracking - Row 1
        self.progress_frame = ttk.LabelFrame(self.main_frame, text="Processing Steps", padding=5)
        self.progress_frame.grid(row=1, column=0, sticky="ew", pady=5)
        
        self.download_frame = self.make_progress_section("1. Downloading ...", "Waiting")
        self.audio_frame = self.make_progress_section("2. Extract Audio and Analyzing audio content...", "Waiting")

        # Results display - Row 2 (will expand)
        self.results_frame = ttk.LabelFrame(self.main_frame, text="Analysis Results", padding=10)
        self.results_frame.grid(row=2, column=0, sticky="nsew", pady=5)
        
        # Configure grid weights for results frame
        self.results_frame.grid_rowconfigure(0, weight=1)
        self.results_frame.grid_columnconfigure(0, weight=1)
        
        self.result_text = tk.Text(self.results_frame, wrap=tk.WORD, state='disabled', padx=5, pady=5)
        self.result_scroll = ttk.Scrollbar(self.results_frame, command=self.result_text.yview)
        self.result_text.config(yscrollcommand=self.result_scroll.set)
        
        # Summary Frame - Row 3
        self.summary_frame = ttk.LabelFrame(self.main_frame, text="Analysis Summary", padding=10)
        self.summary_frame.grid(row=3, column=0, sticky="ew", pady=5)
        
        self.summary_label = ttk.Label(self.summary_frame, text="No analysis performed yet")
        self.summary_label.pack(anchor=tk.NW)
        
        # Buttons Frame - Row 4 (fixed at bottom)
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.grid(row=4, column=0, sticky="ew", pady=5)
        
        self.analyze_button = ttk.Button(self.button_frame, text="Start Analysis", 
                                      command=self.start_analysis, width=15)
        self.analyze_button.pack(side=tk.LEFT, padx=5)
        
        self.reset_button = ttk.Button(self.button_frame, text="Reset", 
                                     command=self.reset_analysis, width=15)
        self.reset_button.pack(side=tk.LEFT, padx=5)
        
    def setup_layout(self):
        # Results frame layout
        self.result_text.grid(row=0, column=0, sticky="nsew")
        self.result_scroll.grid(row=0, column=1, sticky="ns")

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

    def on_url_entry(self, event=None):
        """Triggered when user enters a URL (either by pressing Enter or leaving the field)"""
        new_url = self.url_entry.get().strip()
        if new_url and new_url != self.current_url:
            self.current_url = new_url
            self.start_analysis()

    def start_analysis(self):
        youtube_url = self.url_entry.get().strip()
        if not youtube_url:
            messagebox.showerror("Error", "Please enter a YouTube URL")
            return
            
        self.reset_analysis()
        self.analyze_button.config(state=tk.DISABLED)
        
        # Run analysis in a separate thread to keep the UI responsive
        threading.Thread(target=self.perform_analysis, args=(youtube_url,), daemon=True).start()

    def perform_analysis(self, youtube_url):
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

            # Store results
            self.current_results = {
                'audio_pred': audio_pred,
                'audio_conf': audio_conf
            }

            # Display results
            self.show_results(audio_pred, audio_conf)
            
            # Update summary
            self.update_summary(audio_pred)
            
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
            self.reset_analysis()
        finally:
            self.analyze_button.config(state=tk.NORMAL)

    def update_summary(self, audio_pred):
        summary = "Analysis Summary:\n\n"
        
        if audio_pred:
            summary += f"Audio: {self.get_category_label(audio_pred)}\n"
        else:
            summary += "Audio: No prediction\n"
            
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

    def show_results(self, audio_pred, audio_conf):
        audio_label = self.get_category_label(audio_pred)

        self.result_text.config(state='normal')
        self.result_text.delete(1.0, tk.END)
        
        results = "=== ANALYSIS RESULTS ===\n\n"
        results += f"Audio Analysis:\n"
        results += f"  - Content: {audio_label}\n"
        results += f"  - Confidence: {round(audio_conf, 2)}%\n\n"
        
        self.result_text.insert(tk.END, results)
        self.result_text.config(state='disabled')

    def reset_analysis(self):
        for frame in [self.download_frame, self.audio_frame]:
            frame.progress['value'] = 0
            frame.status.config(text="Waiting", foreground="gray")
        
        self.result_text.config(state='normal')
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "Results will appear here after analysis...")
        self.result_text.config(state='disabled')
        
        if hasattr(self, 'current_results'):
            del self.current_results
        
        # Reset summary
        self.summary_label.config(text="No analysis performed yet")
            
        self.root.update()

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioAnalyzerApp(root)
    root.mainloop()