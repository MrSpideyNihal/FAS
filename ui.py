import cv2
import os
import numpy as np
import csv
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import threading
import time

class FacialAttendanceSystemUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Facial Attendance System")
        self.root.geometry("1000x600")
        self.root.resizable(True, True)
        self.current_subject = ""
        self.current_faculty = ""
        # Initialize backend system
        #no argument passed
        self.system = SimpleFacialAttendanceSystem()
        
        # Variables
        self.is_camera_running = False
        self.camera_thread = None
        self.current_frame = None
        
        # Create the UI
        self._create_ui()
    
    def _create_ui(self):
        """Create the main user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel (camera view and controls)
        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        left_panel.rowconfigure(0, weight=1)  # Camera view row expands
        left_panel.rowconfigure(1, weight=0)  # Controls row remains fixed

        # Camera view frame
        self.camera_frame = ttk.LabelFrame(left_panel, text="Camera View", padding=10)
        self.camera_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        
        # Camera display
        self.camera_label = ttk.Label(self.camera_frame)
        self.camera_label.pack(fill=tk.BOTH, expand=True)
        
        # Camera controls frame
        camera_controls = ttk.Frame(left_panel, padding=10)
        camera_controls.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        
        # Start/Stop Camera button
        self.camera_btn_text = tk.StringVar(value="Start Camera")
        self.camera_btn = ttk.Button(camera_controls, textvariable=self.camera_btn_text, 
                                     command=self.toggle_camera)
        self.camera_btn.pack(side=tk.LEFT, padx=5)
        
        # Capture button (for registration)
        self.capture_btn = ttk.Button(camera_controls, text="Capture", 
                                     command=self.capture_face, state=tk.DISABLED)
        self.capture_btn.pack(side=tk.LEFT, padx=5)
        
        # Right panel (tabbed interface)
        right_panel = ttk.Frame(main_frame, width=300)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5, pady=5)
        
        # Create notebook (tabs)
        self.notebook = ttk.Notebook(right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.registration_tab = ttk.Frame(self.notebook, padding=10)
        self.attendance_tab = ttk.Frame(self.notebook, padding=10)
        self.reports_tab = ttk.Frame(self.notebook, padding=10)
        
        self.notebook.add(self.registration_tab, text="Registration")
        self.notebook.add(self.attendance_tab, text="Attendance")
        self.notebook.add(self.reports_tab, text="Reports")
        
        # Setup each tab
        self._setup_registration_tab()
        self._setup_attendance_tab()
        self._setup_reports_tab()
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def _setup_registration_tab(self):
        """Setup the registration tab"""
        # Name entry
        name_frame = ttk.Frame(self.registration_tab)
        name_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(name_frame, text="Name:").pack(side=tk.LEFT, padx=5)
        self.name_entry = ttk.Entry(name_frame)
        self.name_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Register button
        self.register_btn = ttk.Button(self.registration_tab, text="Start Registration", 
                                      command=self.start_registration)
        self.register_btn.pack(fill=tk.X, pady=10)
        
        # Registration progress
        self.progress_frame = ttk.LabelFrame(self.registration_tab, text="Registration Progress", padding=10)
        self.progress_frame.pack(fill=tk.X, pady=5)
        
        self.progress_var = tk.IntVar(value=0)
        self.progress_bar = ttk.Progressbar(self.progress_frame, orient=tk.HORIZONTAL, 
                                           length=200, mode='determinate', variable=self.progress_var)
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        self.progress_label = ttk.Label(self.progress_frame, text="0/30 images captured")
        self.progress_label.pack(pady=5)
        
        # Registered users list
        users_frame = ttk.LabelFrame(self.registration_tab, text="Registered Users", padding=10)
        users_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Scrollbar and listbox for users
        scrollbar = ttk.Scrollbar(users_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.users_listbox = tk.Listbox(users_frame)
        self.users_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.users_listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.users_listbox.yview)
        
        # Buttons frame for user management
        user_buttons_frame = ttk.Frame(users_frame)
        user_buttons_frame.pack(fill=tk.X, pady=5)
        
        # Refresh users button
        refresh_btn = ttk.Button(user_buttons_frame, text="Refresh List", command=self.refresh_users_list)
        refresh_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        # Delete user button
        delete_btn = ttk.Button(user_buttons_frame, text="Delete User", command=self.delete_registered_user)
        delete_btn.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=2)
        
        # Initialize the list
        self.refresh_users_list()
    def _setup_attendance_tab(self):
        """Setup the attendance tab"""
        # Subject and faculty entry fields
        subject_frame = ttk.Frame(self.attendance_tab)
        subject_frame.pack(fill=tk.X, pady=5)

        ttk.Label(subject_frame, text="Subject:").pack(side=tk.LEFT, padx=5)
        self.subject_entry = ttk.Entry(subject_frame)
        self.subject_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        faculty_frame = ttk.Frame(self.attendance_tab)
        faculty_frame.pack(fill=tk.X, pady=5)

        ttk.Label(faculty_frame, text="Faculty:").pack(side=tk.LEFT, padx=5)
        self.faculty_entry = ttk.Entry(faculty_frame)
        self.faculty_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # Start/Stop attendance button
        self.attendance_btn_text = tk.StringVar(value="Start Attendance")
        self.attendance_btn = ttk.Button(self.attendance_tab, textvariable=self.attendance_btn_text, 
                                        command=self.toggle_attendance)
        self.attendance_btn.pack(fill=tk.X, pady=10)

        # Today's attendance frame
        attendance_frame = ttk.LabelFrame(self.attendance_tab, text="Today's Attendance", padding=10)
        attendance_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Create treeview for attendance
        columns = ("name", "time_in", "time_out", "subject", "faculty")
        self.attendance_tree = ttk.Treeview(attendance_frame, columns=columns, show="headings")

        # Define headings
        self.attendance_tree.heading("name", text="Name")
        self.attendance_tree.heading("time_in", text="Time In")
        self.attendance_tree.heading("time_out", text="Time Out")
        self.attendance_tree.heading("subject", text="Subject")
        self.attendance_tree.heading("faculty", text="Faculty")

        # Set column widths
        self.attendance_tree.column("name", width=120)
        self.attendance_tree.column("time_in", width=80)
        self.attendance_tree.column("time_out", width=80)
        self.attendance_tree.column("subject", width=120)
        self.attendance_tree.column("faculty", width=120)

        # Add scrollbar
        tree_scroll = ttk.Scrollbar(attendance_frame, orient="vertical", command=self.attendance_tree.yview)
        self.attendance_tree.configure(yscrollcommand=tree_scroll.set)

        # Pack everything
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.attendance_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Refresh button
        refresh_btn = ttk.Button(self.attendance_tab, text="Refresh Attendance", 
                               command=self.refresh_attendance)
        refresh_btn.pack(fill=tk.X, pady=5)
    def export_report(self):
        """Export attendance report to a CSV file"""
        date = self.date_entry.get().strip()
        subject = self.report_subject_entry.get().strip()
        faculty = self.report_faculty_entry.get().strip()

        date_str = date if date else "all_dates"
        subject_str = f"_{subject}" if subject else ""
        print(f"subject_str: {subject_str}")
        faculty_str = f"_{faculty}" if faculty else ""

        # Ask for output file
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=f"attendance_report_{date_str}{subject_str}{faculty_str}.csv"
        )

        if filename:
            # Export the report
            records = self.system.export_attendance_report(
                filename, 
                date if date else None,
                subject if subject else None,
                faculty if faculty else None
            )

            if records:
                messagebox.showinfo("Success", f"Report exported to {filename}")
                self.status_var.set(f"Report exported to {filename}")
            else:
                messagebox.showwarning("Warning", "No records found for export")
    def view_report(self, all_dates=False):
        """View attendance report for a specific date or all dates"""
        # Clear current display
        for item in self.report_tree.get_children():
            self.report_tree.delete(item)

        date = None if all_dates else self.date_entry.get().strip()
        subject_filter = self.report_subject_entry.get().strip()
        faculty_filter = self.report_faculty_entry.get().strip()

        try:
            # Read attendance data
            attendance_records = []
            with open(self.system.attendance_file, 'r', newline='') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    # Apply filters
                    matches_date = date is None or not date or row["date"] == date
                    matches_subject = not subject_filter or row.get("subject", "") == subject_filter
                    matches_faculty = not faculty_filter or row.get("faculty", "") == faculty_filter

                    if matches_date and matches_subject and matches_faculty:
                        # Calculate duration if both time_in and time_out exist
                        duration = ""
                        if row["time_in"] and row["time_out"]:
                            try:
                                time_in = datetime.strptime(row["time_in"], "%H:%M:%S")
                                time_out = datetime.strptime(row["time_out"], "%H:%M:%S")
                                duration = str(time_out - time_in)
                            except:
                                pass
                            
                        # Get subject and faculty (may be missing in old records)
                        subject = row.get("subject", "")
                        faculty = row.get("faculty", "")

                        # Add to tree
                        self.report_tree.insert("", tk.END, values=(
                            row["id"], 
                            row["student_name"], 
                            row["date"], 
                            row["time_in"], 
                            row["time_out"], 
                            duration,
                            row["subject"],
                            faculty
                        ))

            # Update status
            filters = []
            if date:
                filters.append(f"date: {date}")
            if subject_filter:
                filters.append(f"subject: {subject_filter}")
            if faculty_filter:
                filters.append(f"faculty: {faculty_filter}")

            if filters:
                self.status_var.set(f"Showing attendance report for {', '.join(filters)}")
            else:
                self.status_var.set("Showing all attendance records")

        except Exception as e:
            messagebox.showerror("Error", f"Could not load attendance data: {e}")
    def _setup_reports_tab(self):
        """Setup the reports tab"""
        # Date selection
        date_frame = ttk.Frame(self.reports_tab)
        date_frame.pack(fill=tk.X, pady=5)

        ttk.Label(date_frame, text="Date (YYYY-MM-DD):").pack(side=tk.LEFT, padx=5)
        self.date_entry = ttk.Entry(date_frame)
        self.date_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.date_entry.insert(0, datetime.now().strftime("%Y-%m-%d"))

        # Subject and Faculty filters
        subject_frame = ttk.Frame(self.reports_tab)
        subject_frame.pack(fill=tk.X, pady=5)

        ttk.Label(subject_frame, text="Subject:").pack(side=tk.LEFT, padx=5)
        self.report_subject_entry = ttk.Entry(subject_frame)
        self.report_subject_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        faculty_frame = ttk.Frame(self.reports_tab)
        faculty_frame.pack(fill=tk.X, pady=5)

        ttk.Label(faculty_frame, text="Faculty:").pack(side=tk.LEFT, padx=5)
        self.report_faculty_entry = ttk.Entry(faculty_frame)
        self.report_faculty_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # View report button
        view_report_btn = ttk.Button(self.reports_tab, text="View Report", 
                                   command=self.view_report)
        view_report_btn.pack(fill=tk.X, pady=5)

        # Export button
        export_btn = ttk.Button(self.reports_tab, text="Export Report", 
                              command=self.export_report)
        export_btn.pack(fill=tk.X, pady=5)

        # Report results frame
        report_frame = ttk.LabelFrame(self.reports_tab, text="Report Results", padding=10)
        report_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Create treeview for report results
        columns = ("id", "name", "date", "time_in", "time_out", "duration", "subject", "faculty")
        self.report_tree = ttk.Treeview(report_frame, columns=columns, show="headings")

        # Define headings
        self.report_tree.heading("id", text="ID")
        self.report_tree.heading("name", text="Name")
        self.report_tree.heading("date", text="Date")
        self.report_tree.heading("time_in", text="Time In")
        self.report_tree.heading("time_out", text="Time Out")
        self.report_tree.heading("duration", text="Duration")
        self.report_tree.heading("subject", text="Subject")
        self.report_tree.heading("faculty", text="Faculty")

        # Set column widths
        self.report_tree.column("id", width=40)
        self.report_tree.column("name", width=120)
        self.report_tree.column("date", width=80)
        self.report_tree.column("time_in", width=70)
        self.report_tree.column("time_out", width=70)
        self.report_tree.column("duration", width=70)
        self.report_tree.column("subject", width=100)
        self.report_tree.column("faculty", width=100)

        # Add scrollbar
        report_scroll = ttk.Scrollbar(report_frame, orient="vertical", command=self.report_tree.yview)
        self.report_tree.configure(yscrollcommand=report_scroll.set)

        # Pack everything
        report_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.report_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # View all button
        view_all_btn = ttk.Button(self.reports_tab, text="View All Records", 
                                command=lambda: self.view_report(all_dates=True))
        view_all_btn.pack(fill=tk.X, pady=5)
    def _setup_data_storage(self):
        """Set up the text file storage system for attendance records."""
        # Create data directories if they don't exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.known_faces_dir, exist_ok=True)

        # Create students CSV file if it doesn't exist
        if not os.path.exists(self.students_file):
            with open(self.students_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['id', 'name', 'registration_date'])

        # Create attendance CSV file if it doesn't exist
        if not os.path.exists(self.attendance_file):
            with open(self.attendance_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['id', 'student_name', 'date', 'time_in', 'time_out', 'subject', 'faculty'])
        else:
            # Check if we need to update the file structure to include subject and faculty
            with open(self.attendance_file, 'r', newline='') as file:
                reader = csv.reader(file)
                header = next(reader, None)
                if header and 'subject' not in header and 'faculty' not in header:
                    # Need to update the file structure
                    records = [header] + list(reader)

                    # Add new columns to header
                    records[0].extend(['subject', 'faculty'])

                    # Add empty values for existing records
                    for i in range(1, len(records)):
                        records[i].extend(['', ''])

                    # Write back the updated file
                    with open(self.attendance_file, 'w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerows(records)

        # Load existing student data
        self._load_students()
    def start_attendance(self):
        """Start attendance mode"""
        if not self.system.loaded_recognizer:   
            messagebox.showerror("Error", "No trained recognizer available. Please register faces first.")
            return

        # Check if subject and faculty are provided
        subject = self.subject_entry.get().strip()
        print("Subject 1st before strip :",subject)
        faculty = self.faculty_entry.get().strip()

        if not subject or not faculty:
            messagebox.showerror("Error", "Please enter both subject and faculty name.")
            return

        # Store current subject and faculty
        self.current_subject = subject
        self.current_faculty = faculty

        # Start camera if not running
        if not self.is_camera_running:
            self.start_camera()

        # Set attendance mode
        self.attendance_running = True
        self.attendance_btn_text.set("Stop Attendance")

        # For tracking who has been marked in this session
        self.marked_this_session = set()

        # Update status
        self.status_var.set(f"Attendance mode active for {subject} by {faculty}. Recognizing faces...")

        # Start attendance thread
        self.attendance_thread = threading.Thread(target=self.attendance_loop)
        self.attendance_thread.daemon = True
        self.attendance_thread.start()
        """Start attendance mode"""
        if not self.system.loaded_recognizer:   
            messagebox.showerror("Error", "No trained recognizer available. Please register faces first.")
            return

        # Check if subject and faculty are provided
        subject = self.subject_entry.get().strip()
        print("subject in entry ",subject)

        faculty = self.faculty_entry.get().strip()

        if not subject or not faculty:
            messagebox.showerror("Error", "Please enter both subject and faculty name.")
            return

        # Store current subject and faculty
        self.current_subject = subject
        self.current_faculty = faculty

        # Start camera if not running
        if not self.is_camera_running:
            self.start_camera()

        # Set attendance mode
        self.attendance_running = True
        self.attendance_btn_text.set("Stop Attendance")

        # For tracking who has been marked in this session
        self.marked_this_session = set()

        # Update status
        self.status_var.set(f"Attendance mode active for {subject} by {faculty}. Recognizing faces...")

        # Start attendance thread
        self.attendance_thread = threading.Thread(target=self.attendance_loop)
        self.attendance_thread.daemon = True
        self.attendance_thread.start()
        """Start attendance mode"""
        if not self.system.loaded_recognizer:   
            messagebox.showerror("Error", "No trained recognizer available. Please register faces first.")
            return

        # Check if subject and faculty are provided
        subject = self.subject_entry.get().strip()
        print("subject in entry ",subject)
        faculty = self.faculty_entry.get().strip()

        if not subject or not faculty:
            messagebox.showerror("Error", "Please enter both subject and faculty name.")
            return

        # Store current subject and faculty
        self.current_subject = subject
        self.current_faculty = faculty

        # Start camera if not running
        if not self.is_camera_running:
            self.start_camera()

        # Set attendance mode
        self.attendance_running = True
        self.attendance_btn_text.set("Stop Attendance")

        # For tracking who has been marked in this session
        self.marked_this_session = set()

        # Update status
        self.status_var.set(f"Attendance mode active for {subject} by {faculty}. Recognizing faces...")

        # Start attendance thread
        self.attendance_thread = threading.Thread(target=self.attendance_loop)
        self.attendance_thread.daemon = True
        self.attendance_thread.start()
    def toggle_camera(self):
        """Toggle camera on/off"""
        if self.is_camera_running:
            self.stop_camera()
        else:
            self.start_camera()
    
    def start_camera(self):
        """Start the camera in a separate thread"""
        if not self.is_camera_running:
            self.is_camera_running = True
            self.camera_btn_text.set("Stop Camera")
            
            # Start camera thread
            self.camera_thread = threading.Thread(target=self.camera_loop)
            self.camera_thread.daemon = True
            self.camera_thread.start()
            
            self.status_var.set("Camera started")
    
    def stop_camera(self):
        """Stop the camera"""
        if self.is_camera_running:
            self.is_camera_running = False
            self.camera_btn_text.set("Start Camera")
            
            # Wait for thread to finish
            if self.camera_thread:
                self.camera_thread.join(timeout=1.0)
            
            # Reset camera label
            self.camera_label.config(image="")
            self.status_var.set("Camera stopped")
    
    def camera_loop(self):
        """Camera processing loop (runs in a separate thread)"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open camera")
            self.is_camera_running = False
            self.camera_btn_text.set("Start Camera")
            return
        
        while self.is_camera_running:
            ret, frame = cap.read()
            if not ret:
                messagebox.showerror("Error", "Failed to grab frame")
                break
            
            # Store current frame (for attendance or registration)
            self.current_frame = frame.copy()
            
            # Display frame
            self.update_camera_display(frame)
        
        cap.release()
    
    def update_camera_display(self, frame):
        """Update the camera display with the current frame"""
        # Convert the frame from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PhotoImage
        img = Image.fromarray(frame_rgb)
        
        # Resize to fit the display
        # Get the current width and height of the camera_label
        width = self.camera_label.winfo_width()
        height = self.camera_label.winfo_height()
        
        # Ensure we have valid dimensions
        if width > 1 and height > 1:
            # Calculate aspect ratio
            img_width, img_height = img.size
            aspect_ratio = img_width / img_height
            
            # Resize while maintaining aspect ratio
            if width / height > aspect_ratio:
                # Height is the limiting factor
                new_height = height
                new_width = int(height * aspect_ratio)
            else:
                # Width is the limiting factor
                new_width = width
                new_height = int(width / aspect_ratio)
            
            img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(image=img)
        
        # Update label
        self.camera_label.config(image=photo)
        self.camera_label.image = photo  # Keep a reference
    
    def start_registration(self):
        """Start the face registration process"""
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showerror("Error", "Please enter a name")
            return
        
        # Enable capture button
        self.capture_btn.config(state=tk.NORMAL)
        
        # Reset progress
        self.captured_count = 0
        self.max_samples = 30
        self.training_data = []
        self.training_labels = []
        
        # Get face ID and create directory
        self.register_name = name
        self.register_face_id = self.system._get_next_face_id()
        
        # Create directory for the person if it doesn't exist
        person_dir = os.path.join(self.system.known_faces_dir, name)
        if not os.path.exists(person_dir):
            os.makedirs(person_dir)
        
        # Update status
        self.status_var.set(f"Registration started for {name}. Capture 30 images.")
        
        # Start camera if not running
        if not self.is_camera_running:
            self.start_camera()
    
    def capture_face(self):
        """Capture a face image for registration"""
        if not self.is_camera_running or self.current_frame is None:
            messagebox.showerror("Error", "Camera is not running")
            return
        
        if not hasattr(self, 'register_name') or not self.register_name:
            messagebox.showerror("Error", "No registration in progress")
            return
        
        # Process the current frame
        frame = self.current_frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.system.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.3, 
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 1:
            x, y, w, h = faces[0]
            face_img = gray[y:y+h, x:x+w]
            
            # Save the image
            person_dir = os.path.join(self.system.known_faces_dir, self.register_name)
            img_path = os.path.join(person_dir, f"{self.register_name}_{self.captured_count}.jpg")
            cv2.imwrite(img_path, face_img)
            
            # Add to training data
            self.training_data.append(face_img)
            self.training_labels.append(self.register_face_id)
            
            # Update count and progress
            self.captured_count += 1
            self.progress_var.set(int(self.captured_count / self.max_samples * 100))
            self.progress_label.config(text=f"{self.captured_count}/{self.max_samples} images captured")
            
            self.status_var.set(f"Captured image {self.captured_count}/{self.max_samples}")
            
            # Check if we've reached the maximum
            if self.captured_count >= self.max_samples:
                self.finish_registration()
        else:
            if len(faces) == 0:
                messagebox.showwarning("Warning", "No face detected. Please try again.")
            else:
                messagebox.showwarning("Warning", "Multiple faces detected. Please ensure only one face is in the frame.")
    def delete_registered_user(self):
        """Delete a registered user"""
        # Get the selected user from the listbox
        selection = self.users_listbox.curselection()
        if not selection:
            messagebox.showerror("Error", "Please select a user to delete")
            return

        # Get the selected user information
        user_info = self.users_listbox.get(selection[0])

        # Extract the name from the user info (format: "id: name (registration_date)")
        # Parse "1: John Doe (2023-04-01)" to get "John Doe"
        try:
            user_id = int(user_info.split(":")[0].strip())
            user_name = user_info.split(":")[1].split("(")[0].strip()
        except:
            messagebox.showerror("Error", "Could not parse user information")
            return

        # Confirm deletion
        confirm = messagebox.askyesno("Confirm Deletion", 
                                     f"Are you sure you want to delete user {user_name}?\n"
                                     "This will remove all face data and cannot be undone.")
        if not confirm:
            return

        # Delete the user from students file
        self.delete_user_from_file(user_id)

        # Delete the user's face directory
        user_dir = os.path.join(self.system.known_faces_dir, user_name)
        if os.path.exists(user_dir) and os.path.isdir(user_dir):
            try:
                for filename in os.listdir(user_dir):
                    file_path = os.path.join(user_dir, filename)
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                os.rmdir(user_dir)
            except Exception as e:
                messagebox.showerror("Error", f"Could not delete user directory: {e}")
                return

        # Remove from face_id_map
        if user_id in self.system.face_id_map:
            del self.system.face_id_map[user_id]

        # Retrain recognizer with remaining users
        self.retrain_recognizer()

        # Refresh the user list
        self.refresh_users_list()

        # Update status
        self.status_var.set(f"User {user_name} deleted successfully")
        messagebox.showinfo("Success", f"User {user_name} deleted successfully")

    def delete_user_from_file(self, user_id):
        """Delete a user from the students CSV file"""
        try:
            # Read all records
            records = []
            with open(self.system.students_file, 'r', newline='') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    # Skip the user to delete
                    if int(row['id']) != user_id:
                        records.append(row)

            # Write back all records
            with open(self.system.students_file, 'w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=['id', 'name', 'registration_date'])
                writer.writeheader()
                writer.writerows(records)

            return True
        except Exception as e:
            messagebox.showerror("Error", f"Could not delete user from file: {e}")
            return False

    def retrain_recognizer(self):
        """Retrain the face recognizer with all remaining registered users"""
        # Train the recognizer with ALL remaining data
        all_faces = []
        all_labels = []

        # Load all registered faces
        for person_name in os.listdir(self.system.known_faces_dir):
            person_dir = os.path.join(self.system.known_faces_dir, person_name)
            if os.path.isdir(person_dir):
                # Get face ID for this person
                person_id = None
                for face_id, name in self.system.face_id_map.items():
                    if name == person_name:
                        person_id = face_id
                        break
                    
                if person_id is not None:
                    for img_name in os.listdir(person_dir):
                        if img_name.endswith(('.jpg', '.jpeg', '.png')):
                            img_path = os.path.join(person_dir, img_name)
                            face_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                            if face_img is not None:
                                all_faces.append(face_img)
                                all_labels.append(person_id)

        if all_faces and all_labels:
            # Train the recognizer
            self.system.recognizer = cv2.face.LBPHFaceRecognizer_create()  # Create a fresh recognizer
            self.system.recognizer.train(all_faces, np.array(all_labels))
            self.system._save_recognizer()
            self.system.loaded_recognizer = True

            # Update status
            self.status_var.set("Face recognizer retrained successfully")
        else:
            # No faces left, just reset the recognizer
            self.system.recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.system.loaded_recognizer = False

            # Delete the recognizer file if it exists
            model_path = os.path.join(self.system.data_dir, "recognizer.yml")
            if os.path.exists(model_path):
                os.remove(model_path)

            # Update status
            self.status_var.set("All users deleted, face recognizer reset") 
    def finish_registration(self):
        """Finish the registration process and train the recognizer"""
        if not hasattr(self, 'register_name') or not self.register_name:
            return

        # Disable capture button
        self.capture_btn.config(state=tk.DISABLED)

        # Update status
        self.status_var.set(f"Training recognizer with {self.captured_count} images...")

        # Add to face ID map
        self.system.face_id_map[self.register_face_id] = self.register_name

        # Add to students file
        self.system._add_student_to_file(self.register_name, self.register_face_id)

        # Train the recognizer with ALL data
        all_faces = []
        all_labels = []

        # ALWAYS load ALL previously registered faces, not just when updating existing model
        # Load all registered faces including the new one
        for person_name in os.listdir(self.system.known_faces_dir):
            person_dir = os.path.join(self.system.known_faces_dir, person_name)
            if os.path.isdir(person_dir):
                # Get face ID for this person
                person_id = None
                for face_id, name in self.system.face_id_map.items():
                    if name == person_name:
                        person_id = face_id
                        break
                    
                if person_id is not None:
                    for img_name in os.listdir(person_dir):
                        if img_name.endswith(('.jpg', '.jpeg', '.png')):
                            img_path = os.path.join(person_dir, img_name)
                            face_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                            if face_img is not None:
                                all_faces.append(face_img)
                                all_labels.append(person_id)

        # No need to add the newly captured faces separately since we're loading ALL faces
        # from directories including the newly registered ones

        if all_faces and all_labels:
            # Train the recognizer
            self.system.recognizer.train(all_faces, np.array(all_labels))
            self.system._save_recognizer()
            self.system.loaded_recognizer = True

            # Update status
            self.status_var.set(f"Successfully registered {self.register_name}")
            messagebox.showinfo("Success", f"Successfully registered {self.register_name} with {self.captured_count} images")

            # Reset progress
            self.progress_var.set(0)
            self.progress_label.config(text="0/30 images captured")

            # Clear name entry
            self.name_entry.delete(0, tk.END)

            # Refresh user list
            self.refresh_users_list()
        else:
            self.status_var.set("No training data available")
            messagebox.showerror("Error", "No training data available")
    def refresh_users_list(self):
        """Refresh the list of registered users"""
        self.users_listbox.delete(0, tk.END)
        
        try:
            with open(self.system.students_file, 'r', newline='') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    self.users_listbox.insert(tk.END, f"{row['id']}: {row['name']} ({row['registration_date']})")
        except Exception as e:
            messagebox.showerror("Error", f"Could not load users: {e}")
    
    def toggle_attendance(self):
        """Toggle attendance mode on/off"""
        if hasattr(self, 'attendance_running') and self.attendance_running:
            self.stop_attendance()
        else:
            self.start_attendance()
    
    def stop_attendance(self):
        """Stop attendance mode"""
        self.attendance_running = False
        self.attendance_btn_text.set("Start Attendance")
        
        # Wait for thread to finish
        if hasattr(self, 'attendance_thread') and self.attendance_thread:
            self.attendance_thread.join(timeout=1.0)
        
        # Update status
        self.status_var.set("Attendance mode stopped")
        
        # Refresh attendance display
        self.refresh_attendance()
    
    def attendance_loop(self):
        """Attendance processing loop (runs in a separate thread)"""
        while self.attendance_running and self.is_camera_running:
            # Process the current frame
            if self.current_frame is not None:
                frame = self.current_frame.copy()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detect faces
                faces = self.system.face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.3, 
                    minNeighbors=5,
                    minSize=(30, 30)
                )

                # Process each detected face
                for (x, y, w, h) in faces:
                    face_img = gray[y:y+h, x:x+w]

                    try:
                        # Use the recognizer to predict who this is
                        face_id, confidence = self.system.recognizer.predict(face_img)

                        # Lower confidence value means better match in LBPH
                        if confidence < 100:
                            name = self.system.face_id_map.get(face_id, "Unknown")
                            confidence_text = f"{round(100 - confidence)}%"
                        else:
                            name = "Unknown"
                            confidence_text = ""

                        # Mark attendance if recognized with good confidence
                        if name != "Unknown" and confidence < 50 and name not in self.marked_this_session:
                            print("Subject:",self.current_subject)
                            self.system.mark_attendance(name, subject=self.current_subject, faculty=self.current_faculty)
                            self.marked_this_session.add(name)

                            # Update status
                            self.status_var.set(f"Marked attendance for {name}")

                            # Refresh attendance display
                            self.refresh_attendance()

                        # Draw rectangle and name on the frame
                        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

                        # Put text below the face
                        text = f"{name} {confidence_text}"
                        cv2.putText(frame, text, (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                    except Exception as e:
                        print(f"Error during recognition: {e}")
                        # Draw red rectangle for error
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

                # Update display with the processed frame
                self.update_camera_display(frame)

            # Small delay to reduce CPU usage
            time.sleep(0.1)
    def refresh_attendance(self):
        """Refresh the attendance display for today"""
        # Clear current display
        for item in self.attendance_tree.get_children():
            self.attendance_tree.delete(item)

        # Get today's date
        current_date = datetime.now().strftime("%Y-%m-%d")

        try:
            # Read attendance data
            with open(self.system.attendance_file, 'r', newline='') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if row["date"] == current_date:
                        # Get subject and faculty from the row
                        subject = row.get("subject", "")
                        faculty = row.get("faculty", "")

                        self.attendance_tree.insert("", tk.END, 
                                                 values=(row["student_name"], 
                                                        row["time_in"], 
                                                        row["time_out"],
                                                        subject,  # Add subject
                                                        faculty)) # Add faculty

        except Exception as e:
            messagebox.showerror("Error", f"Could not load attendance data: {e}")
        """Refresh the attendance display for today"""
        # Clear current display
        for item in self.attendance_tree.get_children():
            self.attendance_tree.delete(item)

        # Get today's date
        current_date = datetime.now().strftime("%Y-%m-%d")

        try:
            # Read attendance data
            with open(self.system.attendance_file, 'r', newline='') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if row["date"] == current_date:
                        # Get subject and faculty (may be missing in old records)
                        subject = row.get("subject", "")
                        faculty = row.get("faculty", "")

                        self.attendance_tree.insert("", tk.END, 
                                                 values=(row["student_name"], 
                                                        row["time_in"], 
                                                        row["time_out"],
                                                        subject,
                                                        faculty))
        except Exception as e:
            messagebox.showerror("Error", f"Could not load attendance data: {e}")

# Keep the original backend class unchanged
class SimpleFacialAttendanceSystem:
    def __init__(self, data_dir="attendance_data", known_faces_dir="known_faces"):
        """Initialize the facial attendance system with text file storage."""
        self.known_faces_dir = known_faces_dir
        self.data_dir = data_dir
        self.students_file = os.path.join(data_dir, "students.csv")
        print
        self.attendance_file = os.path.join(data_dir, "attendance.csv")
        
        # Initialize OpenCV face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize face recognizer
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_id_map = {}  # Maps IDs to names
        self.loaded_recognizer = False
        
        # Setup storage
        self._setup_data_storage()
        
        # Try to load existing recognizer model
        self._load_recognizer()
    
    def _setup_data_storage(self):
        """Set up the text file storage system for attendance records."""
        # Create data directories if they don't exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.known_faces_dir, exist_ok=True)
        
        # Create students CSV file if it doesn't exist
        if not os.path.exists(self.students_file):
            with open(self.students_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['id', 'name', 'registration_date'])
        
        # Create attendance CSV file if it doesn't exist
        if not os.path.exists(self.attendance_file):
            with open(self.attendance_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['id', 'student_name', 'date', 'time_in', 'time_out'])
        
        # Load existing student data
        self._load_students()
    
    def _load_students(self):
        """Load existing student data from CSV file."""
        try:
            with open(self.students_file, 'r', newline='') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    face_id = int(row['id'])
                    name = row['name']
                    self.face_id_map[face_id] = name
        except Exception as e:
            print(f"Error loading students: {e}")
    
    def _get_next_face_id(self):
        """Get the next available face ID."""
        if not self.face_id_map:
            return 1
        return max(self.face_id_map.keys()) + 1
    
    def _add_student_to_file(self, name, face_id):
        """Add a new student to the CSV file."""
        registration_date = datetime.now().strftime("%Y-%m-%d")
        
        with open(self.students_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([face_id, name, registration_date])
    
    def _load_recognizer(self):
        """Load the trained face recognizer if available."""
        model_path = os.path.join(self.data_dir, "recognizer.yml")
        
        if os.path.exists(model_path):
            try:
                self.recognizer.read(model_path)
                self.loaded_recognizer = True
                print("Loaded existing face recognizer model")
            except Exception as e:
                print(f"Error loading recognizer model: {e}")
                self.loaded_recognizer = False
        else:
            print("No existing recognizer model found")
            self.loaded_recognizer = False
    
    def _save_recognizer(self):
        """Save the trained face recognizer model."""
        model_path = os.path.join(self.data_dir, "recognizer.yml")
        self.recognizer.write(model_path)
        print("Saved face recognizer model")
    def mark_attendance(self, name, subject=None, faculty=None):
        """Mark attendance for a recognized student."""
        # Get current date and time
        current_date = datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.now().strftime("%H:%M:%S")

        # Check if student already has an entry for today
        student_found = False
        records = []

        try:
            # Read existing records
            with open(self.attendance_file, 'r', newline='') as file:
                reader = csv.DictReader(file)
                header = reader.fieldnames

                # Check if header includes subject and faculty
                if 'subject' not in header:
                    header.append('subject')
                if 'faculty' not in header:
                    header.append('faculty')

                for row in reader:
                    # Make sure row has subject and faculty keys
                    if 'subject' not in row:
                        row['subject'] = ''
                    if 'faculty' not in row:
                        row['faculty'] = ''

                    # Match by name, date, AND subject (if provided)
                    if (row['student_name'] == name and 
                        row['date'] == current_date and 
                        (subject is None or row['subject'] == subject)):

                        # Student already has an entry for today
                        student_found = True

                        # If time_out is empty, update it
                        if not row['time_out']:
                            row['time_out'] = current_time

                        # Update subject and faculty if they were empty
                        if subject and not row['subject']:
                            row['subject'] = subject
                        if faculty and not row['faculty']:
                            row['faculty'] = faculty

                    records.append(row)

            # If student not found, add a new entry
            if not student_found:
                # Find the student ID
                student_id = None
                for face_id, student_name in self.face_id_map.items():
                    if student_name == name:
                        student_id = face_id
                        break
                    
                if student_id is not None:
                    records.append({
                        'id': student_id,
                        'student_name': name,
                        'date': current_date,
                        'time_in': current_time,
                        'time_out': '',
                        'subject': subject or '',
                        'faculty': faculty or ''
                    })

            # Write back all records
            with open(self.attendance_file, 'w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=['id', 'student_name', 'date', 'time_in', 'time_out', 'subject', 'faculty'])
                writer.writeheader()
                writer.writerows(records)

            print(f"Marked attendance for {name} on {current_date} at {current_time}")
            return True

        except Exception as e:
            print(f"Error marking attendance: {e}")
            return False    
            """Mark attendance for a recognized student."""
            # Get current date and time
            current_date = datetime.now().strftime("%Y-%m-%d")
            current_time = datetime.now().strftime("%H:%M:%S")

            # Check if student already has an entry for today
            student_found = False
            records = []

            try:
                # Read existing records
                with open(self.attendance_file, 'r', newline='') as file:
                    reader = csv.DictReader(file)
                    for row in reader:
                        # Match by name, date, AND subject
                            if (row['student_name'] == name and 
                                row['date'] == current_date and 
                                row.get('subject', '') == (subject or '')):

                                # Student already has an entry for today
                                student_found = True

                                # If time_out is empty, update it
                                if not row['time_out']:
                                    row['time_out'] = current_time

                            records.append(row)

                    # If student not found, add a new entry
                    if not student_found:
                        # Find the student ID
                        student_id = None
                        for face_id, student_name in self.face_id_map.items():
                            if student_name == name:
                                student_id = face_id
                                break
                            
                        if student_id is not None:
                            records.append({
                                'id': student_id,
                                'student_name': name,
                                'date': current_date,
                                'time_in': current_time,
                                'time_out': '',
                                'subject': subject or '',
                                'faculty': faculty or ''
                            })

                    # Write back all records
                    with open(self.attendance_file, 'w', newline='') as file:
                        writer = csv.DictWriter(file, fieldnames=['id', 'student_name', 'date', 'time_in', 'time_out', 'subject', 'faculty'])
                        writer.writeheader()
                        writer.writerows(records)

                    print(f"Marked attendance for {name} on {current_date} at {current_time}")
                    return True

            except Exception as e:
                    print(f"Error marking attendance: {e}")
                    return False
    def export_attendance_report(self, output_file, date=None, subject=None, faculty=None):
        """Export attendance report to a CSV file."""
        try:
            # Read all records
            with open(self.attendance_file, 'r', newline='') as file:
                reader = csv.DictReader(file)
                records = []

                for row in reader:
                    # Filter by date and other criteria if specified
                    match_date = date is None or row['date'] == date
                    match_subject = subject is None or row.get('subject', '') == subject
                    match_faculty = faculty is None or row.get('faculty', '') == faculty

                    if match_date and match_subject and match_faculty:
                        # Calculate duration if both time_in and time_out exist
                        duration = ""
                        if row['time_in'] and row['time_out']:
                            try:
                                time_in = datetime.strptime(row['time_in'], "%H:%M:%S")
                                time_out = datetime.strptime(row['time_out'], "%H:%M:%S")
                                duration = str(time_out - time_in)
                            except Exception as e:
                                print(f"Error calculating duration: {e}")

                        # Add duration to the record
                        row_with_duration = row.copy()
                        row_with_duration['duration'] = duration
                        records.append(row_with_duration)

            # If no records found, return empty
            if not records:
                return []

            # Write to output file
            with open(output_file, 'w', newline='') as file:
                fieldnames = ['id', 'student_name', 'date', 'time_in', 'time_out', 'duration', 'subject', 'faculty']
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(records)

            return records

        except Exception as e:
            print(f"Error exporting attendance report: {e}")
            return []

# To run the application
if __name__ == "__main__":
    # Create root window
    root = tk.Tk()
    
    # Apply a theme (optional)
    try:
        style = ttk.Style()
        style.theme_use("clam")  # You can try different themes: 'clam', 'alt', 'default', 'classic'
    except:
        pass
    
    # Create the app
    app = FacialAttendanceSystemUI(root)
    
    # Start the main loop
    root.mainloop()