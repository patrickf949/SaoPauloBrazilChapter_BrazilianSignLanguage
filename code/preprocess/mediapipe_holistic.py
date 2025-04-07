from typing import Dict, List, Literal, Optional

import cv2
import mediapipe as mp
import numpy as np


class MediaPipeHolistic:
    def __init__(self,
                 static_image_mode: bool = False,
                 model_complexity: int = 1,
                 smooth_landmarks: bool = True,
                 enable_segmentation: bool = False,
                 smooth_segmentation: bool = True,
                 refine_face_landmarks: bool = True,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize MediaPipe Holistic detector for combined face, pose, and hand detection.
        
        Args:
            static_image_mode: Whether to treat input images as a video stream (False) or independent images (True).
                             For video processing, set to False to enable tracking instead of detection per image.
            model_complexity: Complexity of the pose landmark model: 0, 1 or 2. 
                            Landmark accuracy as well as inference latency generally go up with the model complexity.
                            0: Fastest but least accurate
                            1: Good balance of speed and accuracy
                            2: Slowest but most accurate
            smooth_landmarks: Whether to filter landmarks across different input images to reduce jitter.
                            Should be disabled for static images.
            enable_segmentation: Whether to generate segmentation mask.
            smooth_segmentation: Whether to filter segmentation across different input images to reduce jitter.
                               Should be disabled for static images.
            refine_face_landmarks: Whether to further refine the landmark coordinates around the eyes and lips, 
                                 and output additional landmarks around the irises.
            min_detection_confidence: Minimum confidence value ([0.0, 1.0]) for person detection to be considered successful.
            min_tracking_confidence: Minimum confidence value ([0.0, 1.0]) for the pose landmarks to be considered tracked 
                                  successfully. Only used when static_image_mode is False.
        """
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_segmentation,
            smooth_segmentation=smooth_segmentation,
            refine_face_landmarks=refine_face_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )    

    def process_frame(self, frame: np.ndarray, timestamp_ms: int) -> Dict:
        """
        Process a single frame and detect holistic landmarks.
        
        Args:
            frame: Input frame in BGR format
            timestamp_ms: Timestamp of the frame in milliseconds
        
        Returns:
            Dictionary containing detection results with the following keys:
            - face_landmarks: 468 face landmarks if detected
            - pose_landmarks: 33 pose landmarks if detected
            - left_hand_landmarks: 21 left hand landmarks if detected
            - right_hand_landmarks: 21 right hand landmarks if detected
            - timestamp_ms: Frame timestamp
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame with holistic model
        results = self.holistic.process(frame_rgb)  # Pass numpy array directly
        
        # Package results
        return {
            'face_landmarks': results.face_landmarks,  # 468 landmarks
            'pose_landmarks': results.pose_landmarks,  # 33 landmarks
            'left_hand_landmarks': results.left_hand_landmarks,  # 21 landmarks
            'right_hand_landmarks': results.right_hand_landmarks,  # 21 landmarks
            'pose_world_landmarks': results.pose_world_landmarks,  # 3D pose landmarks
            'timestamp_ms': timestamp_ms
        }

    def process_video(self, video_path: str, output_path: Optional[str] = None) -> List[Dict]:
        """
        Process a video file and detect holistic landmarks in each frame.
        No post-processing (filtering, centering, scaling) is applied.
        
        Args:
            video_path: Path to input video file
            output_path: Optional path to save the processed video with visualizations
        
        Returns:
            List of dictionaries containing detection results for each frame
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_time_ms = int(1000 / fps)
        all_results = []
        current_time_ms = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                results = self.process_frame(frame, current_time_ms)
                all_results.append(results)
                
                # Visualize if output path is provided
                if output_path:
                    annotated_frame = self.draw_landmarks_on_frame(frame, results)
                    out.write(annotated_frame)
                
                current_time_ms += frame_time_ms
        
        finally:
            cap.release()
            if output_path:
                out.release()
        
        return all_results
    
    def __del__(self):
        """Clean up resources"""
        self.holistic.close()
    
    ### Post Processing

    def get_frame_horizontal_offset(self, results: Dict, 
                           use_shoulders: bool = True,
                           use_face: bool = True,
                           use_hips: bool = False,
                           face_ref: Literal[
                               "nose_tip", 
                               "mean_point",
                               "pupils",
                               "cheeck_bones",
                               "ears",
                               ] = "nose_tip") -> Dict[str, float]:
        """
        Calculate the horizontal offset for each specified reference point.
        
        Args:
            results: Dictionary containing detection results
            use_shoulders: Whether to calculate shoulder midpoint offset
            use_face: Whether to calculate face center offset
            use_hips: Whether to calculate hip midpoint offset
            face_ref: Reference point for face offset calculation. Options are "nose_tip", "mean_point", "pupils", "cheeck_bones". Default is "nose_tip".
                    - "nose_tip": Use the nose tip landmark for face offset.
                    - "mean_point": Use the mean of the top of head and chin landmarks for face offset.
                    - "pupils": Use the mean of the left and right eye landmarks for face offset.
                    - "cheeck_bones": Use the mean of the left and right cheek landmarks for face offset.
                    - "ears": Use the mean of the left and right ear landmarks for face offset.
        
        Returns:
            Dictionary containing horizontal offsets (0.0 to 1.0, where 0.5 is center) for each enabled reference point:
            {
                'shoulders': float,  # Only if use_shoulders is True
                'face': float,       # Only if use_face is True and face landmarks are detected
                'hips': float        # Only if use_hips is True
            }
        """
        offsets = {}
        
        if not results['pose_landmarks']:
            return offsets
            
        pose_landmarks = results['pose_landmarks'].landmark
        
        if use_shoulders:
            left_shoulder = pose_landmarks[11]
            right_shoulder = pose_landmarks[12]
            offsets['shoulders'] = (left_shoulder.x + right_shoulder.x) / 2
        
        if use_face and results['face_landmarks']:
            face_landmarks = results['face_landmarks'].landmark
            match face_ref:
                case "nose_tip":
                    nose_tip = face_landmarks[5]  # Nose tip
                    offsets['face'] = nose_tip.x
                case "mean_point":
                    top_head = face_landmarks[10]  # Top of head
                    chin = face_landmarks[152]  # Chin
                    offsets['face'] = (top_head.x + chin.x) / 2
                case "pupils":
                    left_pupil = face_landmarks[133]  # Left eye
                    right_pupil = face_landmarks[362]  # Right eye
                    offsets['face'] = (left_pupil.x + right_pupil.x) / 2
                case "cheeck_bones":
                    left_cheek = face_landmarks[234]
                    right_cheek = face_landmarks[454]
                    offsets['face'] = (left_cheek.x + right_cheek.x) / 2
                case "ears":
                    left_ear = face_landmarks[234]
                    right_ear = face_landmarks[454]
                    offsets['face'] = (left_ear.x + right_ear.x) / 2
                case _:
                    raise ValueError("Invalid face_ref value. Choose from 'nose_tip', 'mean_point', 'pupils', 'cheeck_.")
        
        if use_hips:
            left_hip = pose_landmarks[23]
            right_hip = pose_landmarks[24]
            offsets['hips'] = (left_hip.x + right_hip.x) / 2
        
        return offsets

    def get_frame_vertical_offset(self, results: Dict, 
                           use_shoulders: bool = True,
                           use_face: bool = True,
                           use_hips: bool = False,
                           face_ref: Literal[
                               "nose_tip",
                               "mean_point",
                               ] = "nose_tip") -> Dict[str, float]:
        """
        Calculate the vertical offset for each specified reference point.
        
        Args:
            results: Dictionary containing detection results
            use_shoulders: Whether to calculate shoulder midpoint offset
            use_face: Whether to calculate face center offset
            use_hips: Whether to calculate hip midpoint offset
            face_ref: Reference point for face offset calculation. Options are "nose_tip" or "mean_point". Default is "nose_tip".
                    - "nose_tip": Use the nose tip landmark for face offset.  
                    - "mean_point": Use the mean of the top of head and chin landmarks for face offset.
        
        Returns:
            Dictionary containing vertical offsets (0.0 to 1.0, where 0.5 is center) for each enabled reference point:
            {
                'shoulders': float,  # Only if use_shoulders is True
                'face': float,       # Only if use_face is True and face landmarks are detected
                'hips': float        # Only if use_hips is True
            }
        """
        offsets = {}
        
        if not results['pose_landmarks']:
            return offsets
            
        pose_landmarks = results['pose_landmarks'].landmark
        
        if use_shoulders:
            left_shoulder = pose_landmarks[11]
            right_shoulder = pose_landmarks[12]
            offsets['shoulders'] = (left_shoulder.y + right_shoulder.y) / 2
        
        if use_face and results['face_landmarks']:
            face_landmarks = results['face_landmarks'].landmark
            match face_ref:
                case "nose_tip":
                    nose_tip = face_landmarks[5]  # Nose tip
                    offsets['face'] = nose_tip.y
                case "mean_point":
                    top_head = face_landmarks[10]  # Top of head
                    chin = face_landmarks[152]  # Chin
                    offsets['face'] = (top_head.y + chin.y) / 2
                case _:
                    raise ValueError("Invalid face_ref value. Choose from 'nose_tip', 'mean_point'.")
        
        if use_hips:
            left_hip = pose_landmarks[23]
            right_hip = pose_landmarks[24]
            offsets['hips'] = (left_hip.y + right_hip.y) / 2
        
        return offsets

    def get_frame_landmark_measurements(self, results: Dict) -> Dict[str, float]:
        """
        Calculate various measurements between landmarks.
        
        Args:
            results: Dictionary containing detection results
        
        Returns:
            Dictionary of measurements:
            - shoulder_width: Distance between left and right shoulders
            - face_width: Approximate face width (distance between left and right face edges)
            - face_height: Approximate face height (distance between top of head and chin)
            - left_arm_length: Distance from left shoulder to left wrist
            - right_arm_length: Distance from right shoulder to right wrist
            - hip_width: Distance between left and right hips
            - shoulder_to_hip: Distance from shoulder midpoint to hip midpoint
            - top_head_to_shoulders: Vertical distance from top of head to shoulder midpoint
            - nose_to_shoulders: Vertical distance from nose tip to shoulder midpoint
            - chin_to_shoulders: Vertical distance from chin to shoulder midpoint
        """
        measurements = {}
        
        if not results['pose_landmarks']:
            return measurements
            
        pose_landmarks = results['pose_landmarks'].landmark
        
        # Get shoulder width and midpoint
        left_shoulder = pose_landmarks[11]
        right_shoulder = pose_landmarks[12]
        shoulder_mid = np.array([
            (left_shoulder.x + right_shoulder.x) / 2,
            (left_shoulder.y + right_shoulder.y) / 2
        ])
        measurements['shoulder_width'] = np.sqrt(
            (left_shoulder.x - right_shoulder.x)**2 + 
            (left_shoulder.y - right_shoulder.y)**2
        )
        
        # Get hip width
        left_hip = pose_landmarks[23]
        right_hip = pose_landmarks[24]
        measurements['hip_width'] = np.sqrt(
            (left_hip.x - right_hip.x)**2 + 
            (left_hip.y - right_hip.y)**2
        )
        
        # Get shoulder to hip distance
        hip_mid = np.array([
            (left_hip.x + right_hip.x) / 2,
            (left_hip.y + right_hip.y) / 2
        ])
        measurements['shoulder_to_hip'] = np.linalg.norm(shoulder_mid - hip_mid)
        
        # Get arm lengths
        left_wrist = pose_landmarks[15]
        right_wrist = pose_landmarks[16]
        measurements['left_arm_length'] = np.sqrt(
            (left_shoulder.x - left_wrist.x)**2 + 
            (left_shoulder.y - left_wrist.y)**2
        )
        measurements['right_arm_length'] = np.sqrt(
            (right_shoulder.x - right_wrist.x)**2 + 
            (right_shoulder.y - right_wrist.y)**2
        )
        
        # Get face measurements if available
        if results['face_landmarks']:
            face_landmarks = results['face_landmarks'].landmark
            
            # Face width (distance between left and right face edges)
            left_face = face_landmarks[234]  # Left face edge
            right_face = face_landmarks[454]  # Right face edge
            measurements['face_width'] = np.sqrt(
                (left_face.x - right_face.x)**2 + 
                (left_face.y - right_face.y)**2
            )
            
            # Face height (distance between top of head and chin)
            top_head = face_landmarks[10]  # Top of head
            chin = face_landmarks[152]  # Chin
            measurements['face_height'] = np.sqrt(
                (top_head.x - chin.x)**2 + 
                (top_head.y - chin.y)**2
            )
            
            # Vertical measurements from face to shoulder midpoint
            nose_tip = face_landmarks[5]  # Nose tip
            
            # Calculate vertical distances (only y-coordinate difference)
            measurements['top_head_to_shoulders'] = abs(top_head.y - shoulder_mid[1])
            measurements['nose_to_shoulders'] = abs(nose_tip.y - shoulder_mid[1])
            measurements['chin_to_shoulders'] = abs(chin.y - shoulder_mid[1])
        
        return measurements

    def apply_frame_analysis_to_video(self, results_list: List[Dict], frame_processing_function, **kwargs) -> Dict[str, Dict[str, float]]:
        """
        Generalized method to calculate statistics for a given frame processing function across multiple frames.
        
        Args:
            results_list: List of dictionaries containing detection results for each frame.
            frame_processing_function: Function to process a single frame and return measurements.
            **kwargs: Additional arguments to pass to the frame processing function.
        
        Returns:
            Dictionary containing statistics for each measurement:
            {
                'measurement_name': {
                    'mean': float,
                    'median': float,
                    'max': float,
                    'min': float
                }
            }
        """
        # Initialize storage for measurements
        all_measurements = {}

        # Process each frame
        for results in results_list:
            frame_measurements = frame_processing_function(results, **kwargs)
            
            # Store measurements
            for name, value in frame_measurements.items():
                if name not in all_measurements:
                    all_measurements[name] = []
                all_measurements[name].append(value)
        
        # Calculate statistics
        stats = {}
        for name, values in all_measurements.items():
            if values:  # Only calculate if we have measurements
                stats[name] = {
                    'mean': np.mean(values),
                    'median': np.median(values),
                    'max': np.max(values),
                    'min': np.min(values)
                }
        
        return stats

    def get_video_horizontal_offsets(self, results_list: List[Dict], use_shoulders: bool = True, use_face: bool = True, use_hips: bool = False) -> Dict[str, Dict[str, float]]:
        """
        Calculate statistics for horizontal offsets across multiple frames.
        """
        return self.apply_frame_analysis_to_video(
            results_list,
            self.get_frame_horizontal_offset,
            use_shoulders=use_shoulders,
            use_face=use_face,
            use_hips=use_hips
        )

    def get_video_vertical_offsets(self, results_list: List[Dict], use_shoulders: bool = True, use_face: bool = True, use_hips: bool = False) -> Dict[str, Dict[str, float]]:
        """
        Calculate statistics for vertical offsets across multiple frames.
        """
        return self.apply_frame_analysis_to_video(
            results_list,
            self.get_frame_vertical_offset,
            use_shoulders=use_shoulders,
            use_face=use_face,
            use_hips=use_hips
        )

    def get_video_landmark_measurements(self, results_list: List[Dict]) -> Dict[str, Dict[str, float]]:
        """
        Calculate statistics for all measurements across multiple frames.
        """
        return self.apply_frame_analysis_to_video(
            results_list,
            self.get_frame_landmark_measurements
        )