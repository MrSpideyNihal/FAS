import cv2
import os
import numpy as np
import csv
from datetime import datetime

class SimpleFacialAttendanceSystem:
    def __init__(self, data_dir="attendance_data", known_faces_dir="known_faces"):
        """Initialize the facial attendance system with text file storage."""
        self.known_faces_dir = known_faces_dir
        self.data_dir = data_dir
        self.students_file = os.path.join(data_dir, "students.csv")
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
        # Create data directory if it doesn't exist
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"Created directory for data: {self.data_dir}")
        
        # Create known faces directory if it doesn't exist
        if not os.path.exists(self.known_faces_dir):
            os.makedirs(self.known_faces_dir)
            print(f"Created directory for known faces: {self.known_faces_dir}")
        
        # Create students.csv if it doesn't exist
        if not os.path.exists(self.students_file):
            with open(self.students_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["id", "name", "registration_date"])
            print(f"Created students file: {self.students_file}")
        
        # Create attendance.csv if it doesn't exist
        if not os.path.exists(self.attendance_file):
            with open(self.attendance_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["id", "student_name", "date", "time_in", "time_out"])
            print(f"Created attendance file: {self.attendance_file}")
    
    def _load_recognizer(self):
        """Load trained recognizer if available."""
        model_path = os.path.join(self.data_dir, "recognizer_model.yml")
        id_map_path = os.path.join(self.data_dir, "face_id_map.csv")
        
        if os.path.exists(model_path) and os.path.exists(id_map_path):
            try:
                # Load the recognizer model
                self.recognizer.read(model_path)
                
                # Load the ID to name mapping
                with open(id_map_path, 'r', newline='') as file:
                    reader = csv.reader(file)
                    next(reader)  # Skip header
                    for row in reader:
                        face_id = int(row[0])
                        name = row[1]
                        self.face_id_map[face_id] = name
                
                self.loaded_recognizer = True
                print(f"Loaded recognizer model with {len(self.face_id_map)} faces")
                return True
            except Exception as e:
                print(f"Error loading recognizer model: {e}")
        
        print("No recognizer model found or failed to load")
        return False
    
    def _save_recognizer(self):
        """Save the trained recognizer model."""
        model_path = os.path.join(self.data_dir, "recognizer_model.yml")
        id_map_path = os.path.join(self.data_dir, "face_id_map.csv")
        
        # Save the model
        self.recognizer.write(model_path)
        
        # Save the ID mapping
        with open(id_map_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["face_id", "name"])
            for face_id, name in self.face_id_map.items():
                writer.writerow([face_id, name])
        
        print(f"Saved recognizer model with {len(self.face_id_map)} faces")
    
    def _get_next_face_id(self):
        """Get the next available face ID."""
        if not self.face_id_map:
            return 1
        return max(self.face_id_map.keys()) 
    
    def _add_student_to_file(self, name, face_id):
        """Add a student to the students.csv file if not already present."""
        # Read existing students
        students = []
        try:
            with open(self.students_file, 'r', newline='') as file:
                reader = csv.DictReader(file)
                students = list(reader)
        except Exception as e:
            print(f"Error reading students file: {e}")
            # If file doesn't exist or is corrupted, create new
            with open(self.students_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["id", "name", "registration_date"])
        
        # Check if student already exists
        if not any(student["name"] == name for student in students):
            # Add student to list
            current_date = datetime.now().strftime("%Y-%m-%d")
            students.append({
                "id": str(face_id),
                "name": name,
                "registration_date": current_date
            })
            
            # Write updated list back to file
            with open(self.students_file, 'w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=["id", "name", "registration_date"])
                writer.writeheader()
                writer.writerows(students)
            
            print(f"Added {name} to students file with ID {face_id}")
    
    def register_new_face(self, name):
        """Register a new face for the attendance system."""
        # Create directory for the person if it doesn't exist
        person_dir = os.path.join(self.known_faces_dir, name)
        if not os.path.exists(person_dir):
            os.makedirs(person_dir)
        
        # Get next available face ID
        face_id = self._get_next_face_id()
        self.face_id_map[face_id] = name
        
        # Add to students file
        self._add_student_to_file(name, face_id)
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return False
        
        print(f"Registering new face for {name} (ID: {face_id})...")
        print("Press 'c' to capture (taking 30 images for training)...")
        print("Press 'q' to quit registration at any time")
        
        # Collect face samples
        count = 0
        max_samples = 30
        training_data = []
        training_labels = []
        
        while count < max_samples:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.3, 
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # Draw rectangles around detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow('Register New Face', frame)
            
            key = cv2.waitKey(1)
            
            # Press 'c' to capture
            if key == ord('c'):
                if len(faces) == 1:
                    x, y, w, h = faces[0]
                    face_img = gray[y:y+h, x:x+w]
                    
                    # Save the image
                    img_path = os.path.join(person_dir, f"{name}_{count}.jpg")
                    cv2.imwrite(img_path, face_img)
                    
                    # Add to training data
                    training_data.append(face_img)
                    training_labels.append(face_id)
                    
                    count += 1
                    print(f"Captured image {count}/{max_samples}")
                else:
                    if len(faces) == 0:
                        print("No face detected. Please try again.")
                    else:
                        print("Multiple faces detected. Please ensure only one face is in the frame.")
            
            # Press 'q' to quit
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if count > 0:
            print(f"Captured {count} images. Training the recognizer...")
            
            # Train recognizer with all available images
            all_faces = []
            all_labels = []
            
            if self.loaded_recognizer:
                # If we're updating an existing model, we need to retrain from scratch
                # This is a limitation of OpenCV's LBPH recognizer
                for person_name in os.listdir(self.known_faces_dir):
                    person_dir = os.path.join(self.known_faces_dir, person_name)
                    if os.path.isdir(person_dir):
                        # Get face ID for this person
                        person_id = None
                        for face_id, name in self.face_id_map.items():
                            if name == person_name and person_name != name:  # Skip the new person
                                person_id = face_id
                                break
                        
                        if person_id:
                            for img_name in os.listdir(person_dir):
                                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                                    img_path = os.path.join(person_dir, img_name)
                                    face_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                                    if face_img is not None:
                                        all_faces.append(face_img)
                                        all_labels.append(person_id)
            
            # Add the newly captured faces
            all_faces.extend(training_data)
            all_labels.extend(training_labels)
            
            if all_faces and all_labels:
                # Train the recognizer
                self.recognizer.train(all_faces, np.array(all_labels))
                self._save_recognizer()
                self.loaded_recognizer = True
                print(f"Successfully registered {name} with {count} images")
                return True
            else:
                print("No training data available")
                return False
        else:
            print(f"Failed to register {name}")
            return False
    
    def mark_attendance(self, name):
        """Mark attendance for a recognized person in the text file."""
        # Read existing attendance records
        attendance_records = []
        try:
            with open(self.attendance_file, 'r', newline='') as file:
                reader = csv.DictReader(file)
                attendance_records = list(reader)
        except Exception as e:
            print(f"Error reading attendance file: {e}")
            # If file doesn't exist or is corrupted, create new
            with open(self.attendance_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["id", "student_name", "date", "time_in", "time_out"])
        
        current_date = datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.now().strftime("%H:%M:%S")
        
        # Check if already marked for today
        today_record = None
        for record in attendance_records:
            if record["student_name"] == name and record["date"] == current_date:
                today_record = record
                break
        
        # Generate new attendance ID
        new_id = 1
        if attendance_records:
            try:
                new_id = max(int(record["id"]) for record in attendance_records) + 1
            except ValueError:
                new_id = len(attendance_records) + 1
        
        if today_record:
            # Update time_out
            today_record["time_out"] = current_time
            print(f"Updated exit time for {name} at {current_time}")
        else:
            # Add new attendance record
            attendance_records.append({
                "id": str(new_id),
                "student_name": name,
                "date": current_date,
                "time_in": current_time,
                "time_out": ""
            })
            print(f"Marked attendance for {name} at {current_time}")
        
        # Write updated records back to file
        with open(self.attendance_file, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=["id", "student_name", "date", "time_in", "time_out"])
            writer.writeheader()
            writer.writerows(attendance_records)
    
    def run_attendance(self):
        """Run the facial attendance system."""
        if not self.loaded_recognizer:
            print("No trained recognizer available. Please register faces first.")
            return
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return
        
        print("Starting facial attendance system...")
        print("Press 'q' to quit")
        
        # For tracking who has been marked in this session
        marked_this_session = set()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
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
                    face_id, confidence = self.recognizer.predict(face_img)
                    
                    # Lower confidence value means better match in LBPH
                    if confidence < 100:  # Adjust this threshold as needed
                        name = self.face_id_map.get(face_id, "Unknown")
                        confidence_text = f"{round(100 - confidence)}%"
                    else:
                        name = "Unknown"
                        confidence_text = ""
                    
                    # Mark attendance if recognized with good confidence
                    if name != "Unknown" and confidence < 70 and name not in marked_this_session:
                        self.mark_attendance(name)
                        marked_this_session.add(name)
                    
                    # Draw rectangle and name
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    
                    # Put text below the face
                    text = f"{name} {confidence_text}"
                    cv2.putText(frame, text, (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                except Exception as e:
                    print(f"Error during recognition: {e}")
                    # Draw red rectangle for error
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            
            # Display the result
            cv2.imshow('Facial Attendance System', frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def export_attendance_report(self, output_file="attendance_report.csv", date=None):
        """Export attendance report for a specific date or all dates."""
        try:
            # Read attendance data
            attendance_records = []
            with open(self.attendance_file, 'r', newline='') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    # Filter by date if specified
                    if date is None or row["date"] == date:
                        # Calculate duration if both time_in and time_out exist
                        duration = None
                        if row["time_in"] and row["time_out"]:
                            try:
                                time_in = datetime.strptime(row["time_in"], "%H:%M:%S")
                                time_out = datetime.strptime(row["time_out"], "%H:%M:%S")
                                duration = str(time_out - time_in)
                            except:
                                pass
                        
                        # Add duration to the record
                        row["duration"] = duration if duration else ""
                        attendance_records.append(row)
            
            # Write to output file
            if attendance_records:
                with open(output_file, 'w', newline='') as file:
                    fieldnames = ["id", "student_name", "date", "time_in", "time_out", "duration"]
                    writer = csv.DictWriter(file, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(attendance_records)
                print(f"Attendance report exported to {output_file}")
            else:
                print("No attendance records found for the specified date" if date else "No attendance records found")
            
            return attendance_records
        
        except Exception as e:
            print(f"Error exporting attendance report: {e}")
            return None


# Example usage
if __name__ == "__main__":
    system = SimpleFacialAttendanceSystem()
    
    while True:
        print("\nFacial Attendance System")
        print("1. Register new face")
        print("2. Start attendance")
        print("3. Export attendance report")
        print("4. Exit")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == '1':
            name = input("Enter person's name: ")
            system.register_new_face(name)
        
        elif choice == '2':
            system.run_attendance()
        
        elif choice == '3':
            date_input = input("Enter date (YYYY-MM-DD) or leave blank for all dates: ")
            date = date_input if date_input else None
            filename = input("Enter output filename (default: attendance_report.csv): ")
            filename = filename if filename else "attendance_report.csv"
            system.export_attendance_report(filename, date)
        
        elif choice == '4':
            print("Exiting program...")
            break
        
        else:
            print("Invalid choice. Please try again.")