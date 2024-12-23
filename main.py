import json
import time
import shutil
import pathlib
import os
import yaml
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from contextlib import contextmanager

import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F
import cv2
import numpy as np
from tqdm import tqdm
import torch
from ultralytics import YOLO

"""create a mongoDB Conection and connect"""

"""Press q to end the CV2"""

@dataclass
class ModelConfig:
    """Configuration settings for the YOLO model."""
    img_size: int  # Size of input images for the model
    batch_size: int  # Batch size for training
    epochs: int  # Number of training epochs
    conf_thres: float  # Confidence threshold for object detection
    iou_thres: float  # Intersection over Union threshold for non-maximum suppression
    pretrained_model: str = "yolov8s.pt"  # Path or name of the pretrained model
    
class DetectionSystem:
    """
    A system for object detection using YOLO models. 
    Handles dataset preparation, model training, evaluation, and real-time processing.
    """

    def __init__(
        self,
        dataset_name: str,
        classes: List[str] = ["Cat", "Dog", "Bird", "Man", "Woman", "Human face"],
        config: Optional[ModelConfig] = None,
        root_dir: Optional[pathlib.Path] = None,
    ):
        """
        Initialize the DetectionSystem.

        Args:
            dataset_name: Name of the dataset.
            classes: List of classes for detection.
            config: Configuration for the model.
            root_dir: Root directory for the project.
        """
        self.dataset_name = dataset_name
        self.classes = classes
        self.config = config or ModelConfig(
            img_size=640,
            batch_size=16,
            epochs=3,
            conf_thres=0.5,
            iou_thres=0.45
        )
        
        # Check if classes are provided, if not, raise an error
        if not self.classes:
            raise ValueError("Classes must be provided during initialization.")
        print(f"Classes provided during initialization: {self.classes}")

        
        # Set up directory structure
        self.root_dir = pathlib.Path(root_dir or "detection_project")
        self.directories = self._initialize_directories()
        self._create_data_yaml()
        
        # Configure device (GPU or CPU)
        self.device = self._setup_device()

    def train_model(self) -> YOLO:
        """
        Train the YOLO model.

        Returns:
            The trained YOLO model.
        """
        model = YOLO(self.config.pretrained_model)  # Load pretrained model
        model.train(
            data=self.data_yaml_path,
            imgsz=self.config.img_size,
            epochs=self.config.epochs,
            batch=self.config.batch_size,
            conf=self.config.conf_thres,
            iou=self.config.iou_thres,
            device=self.device
        )

        return model

    @staticmethod
    def _setup_device() -> torch.device:
        """Set up and configure the processing device (GPU/CPU)."""
        if not torch.cuda.is_available():
            print("WARNING: CUDA is not available. Using CPU instead.")
            return torch.device("cpu")
            
        device = torch.device("cuda:0")
        torch.backends.cudnn.benchmark = True  # Optimize for fixed input size
        torch.backends.cudnn.deterministic = False # Optimize for speed
        torch.cuda.empty_cache()  # Free unused GPU memory
        
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

        return device

    def _initialize_directories(self) -> Dict[str, pathlib.Path]:
        """Initialize and create project directories."""
        directories = {
            "models": self.root_dir / "models",  # Directory for storing trained models
            "data": self.root_dir / "data",  # Main data directory
            "raw_data": self.root_dir / "data" / "raw_data",  # Directory for raw data
            "processed_data": self.root_dir / "data" / "processed_data",  # Directory for processed data
            "yolo_data": self.root_dir / "data" / "yolo_format",  # Directory for YOLO format data
            "train": self.root_dir / "data" / "yolo_format" / "train",  # Training data directory
            "val": self.root_dir / "data" / "yolo_format" / "val",  # Validation data directory
            "test": self.root_dir / "data" / "yolo_format" / "test",  # Test data directory
            "results": self.root_dir / "results",  # Directory for storing results
            "logs": self.root_dir / "logs"  # Directory for storing logs
        }

        for dir_path in directories.values():
            dir_path.mkdir(parents=True, exist_ok=True)  # Create directories

        self._clear_yolo_directories(directories)  # Clear YOLO data directories
        return directories

    @staticmethod
    def _clear_yolo_directories(directories: Dict[str, pathlib.Path]) -> None:
        """Clear YOLO data directories and create necessary subdirectories."""
        for split in ["train", "val", "test"]:
            split_dir = directories[split]
            for subdir in ["images", "labels"]:
                subdir_path = split_dir / subdir
                if subdir_path.exists():
                    shutil.rmtree(subdir_path)  # Ensure previous data is removed
                subdir_path.mkdir(parents=True)  # Create subdirectories

            label_cache = split_dir / "labels.cache"
            if label_cache.exists():
                os.remove(str(label_cache))  # Remove label cache file
    

    def _create_data_yaml(self) -> None:
        """Create YAML configuration for YOLO training."""
        yaml_content = {
            "train": str(self.directories["train"].resolve() / "images"),  # Path to training images
            "val": str(self.directories["val"].resolve() / "images"),  # Path to validation images
            "test": str(self.directories["test"].resolve() / "images"),  # Path to test images
            "nc": len(self.classes),  # Number of classes
            "names": self.classes  # List of class names
        }

        yaml_path = self.directories["data"] / "data.yaml"  # Path to data.yaml
        with open(yaml_path, "w") as f:
            yaml.safe_dump(yaml_content, f)  # Write YAML content to file

        self.data_yaml_path = yaml_path  # Store path to data.yaml

    @contextmanager
    def _dataset_context(self, dataset_name: str):
        """Context manager for dataset handling with proper cleanup."""
        try:
            yield  # Yield control to the caller
        except Exception as e:
            if fo.dataset_exists(dataset_name):
                fo.delete_dataset(dataset_name)  # Delete dataset if an error occurs
            raise e

    def _create_splits(
        self, 
        dataset: fo.Dataset, 
        split_ratios: List[float] = [0.8, 0.1, 0.1]
    ) -> Tuple[fo.Dataset, fo.Dataset, fo.Dataset]:
        """
        Create train, validation, and test splits from the dataset.
        
        Args:
            dataset: The dataset to split
            split_ratios: List of ratios for [train, val, test] splits
            
        Returns:
            Tuple of (train_split, val_split, test_split)
        """
        if abs(sum(split_ratios) - 1.0) > 1e-5:  # Robust check for ratios
            raise ValueError("Split ratios must sum to 1")

        # Ensure dataset view for balanced class distribution, 
        dataset.shuffle(seed=51)  # Shuffle dataset for random splits

        train_size = int(split_ratios[0] * len(dataset))  # Calculate training set size
        val_size = int(split_ratios[1] * len(dataset))  # Calculate validation set size

        train_view = dataset.take(train_size)  # Create training view
        val_view = dataset.skip(train_size).take(val_size)  # Create validation view
        test_view = dataset.skip(train_size + val_size)  # Create test view

        train_split = train_view.clone()  # Create training split

        val_split = val_view.clone()  # Create validation split
        test_split = test_view.clone()  # Create test split



        # Add tags to the splits
        train_split.tag_samples("train")
        val_split.tag_samples("val")
        test_split.tag_samples("test")

        return train_split, val_split, test_split

    def _export_splits(
        self, 
        dataset: fo.Dataset, 
        splits: List[fo.Dataset]
    ) -> None:
        """
        Export the splits to YOLO format.
        
        Args:
            dataset: The original dataset
            splits: List of [train_split, val_split, test_split]
        """
        split_names = ["train", "val", "test"]

        for split, split_name in zip(splits, split_names):
            dir_path = self.directories[split_name]
            print(f"Exporting {split_name} split to {dir_path}")

            split.export(
                export_dir=str(dir_path / "images"),  # Export images to the corresponding directory
                dataset_type=fo.types.YOLOv5Dataset,  # Export in YOLOv5 format
                label_field="ground_truth",  # Specify the label field
                classes=self.classes,  # Provide explicit classes to be exported

            )

    def prepare_dataset(
            self,
            confidence_thresh: float = 0.04,
            area_thresh: float = 0.01,
            samples_per_class: int = 600

        ) -> Tuple[fo.Dataset, Dict[str, int]]:
            """
            Prepare and process the dataset with balanced class distribution using Best-First Search.

            Args:
                confidence_thresh: Confidence threshold for filtering detections.
                area_thresh: Area threshold for filtering detections.
                samples_per_class: Number of samples per class.

            Returns:
                Tuple of (merged_dataset, class_distribution).
            """
            dataset_name = f"{self.dataset_name}--{int(time.time())}"
            self.dataset_name = dataset_name  # store the dataset name as a class property for later access
            with self._dataset_context(dataset_name):  # Handle cleanup upon exception
                merged_dataset = fo.Dataset(dataset_name)  # Persistent to avoid duplicates
                merged_dataset.persistent = True

                # Process each class with sample limit

                for cls in self.classes:
                    class_view = foz.load_zoo_dataset(
                        "open-images-v7",
                        split="validation",
                        label_types=["detections"],
                        classes=[cls],  # Filter by class
                        seed=51,  # Seed to guarantee same sampling
                        shuffle=True,
                        max_samples=samples_per_class
                    ).clone()
                    
                    # ------------------------------------------------
                    # Implementation of Best-First Search Algorithm Begins
                    # Here, we process all classes concurrently using a priority queue. The priority is based
                    # on the remaining samples needed per class to achieve balanced selection.
                    # ------------------------------------------------
                    
                    import heapq  # For the priority queue in Best-First Search
                    
                    queue = []  # Initialize queue

                    explored = set() # Track all views that have been explored
                    max_samples_per_class = samples_per_class # store the max samples to be taken per class

                    for cls in self.classes:
                       initial_view = foz.load_zoo_dataset(
                           "open-images-v7",
                           split="validation",
                            label_types=["detections"],
                            classes=[cls],
                            seed=51,  # Seed to guarantee same sampling
                            shuffle=True,
                       ) 
                       
                       heapq.heappush(queue, (0, 0, cls, initial_view)) # Initialize queue with tuple of (cost, samples_added, class, view)

                    
                    
                    with tqdm(total=len(self.classes) * samples_per_class, desc="Processing all classes") as pbar:
                        while queue and len(merged_dataset) < len(self.classes) * samples_per_class:
                            cost, samples_added, cls, current_view = heapq.heappop(queue)  # Best First search, pops element based on cost
                        
                            if len(merged_dataset) >= len(self.classes)*samples_per_class or str(current_view) in explored:
                                continue  # if the length limit is reached, then skip or this view has been explored before
                            explored.add(str(current_view)) # add current view to explored

                            merged_dataset.merge_samples(current_view.take(1))  # Process a single sample from each view
                            pbar.update(1) # Increment the progress bar

                            remaining_view = current_view.skip(1) # Get remaining
                                                
                            if len(remaining_view) > 0:

                                # Calculate heuristic for priority (remaining samples needed)
                                heuristic = max_samples_per_class - (len(merged_dataset) // len(self.classes))
                                
                                heapq.heappush(queue, (heuristic, samples_added+1, cls, remaining_view )) # Prioritize based on heuristic and push to the queue

                    
                    # ------------------------------------------------
                    # Implementation of Best-First Search Algorithm Ends
                    # ------------------------------------------------
                    # merged_dataset.merge_samples(class_view)
                

                train_split, val_split, test_split = self._create_splits(merged_dataset)  # Create dataset splits

                self._export_splits(merged_dataset, [train_split, val_split, test_split])  # Export splits to YOLO format
                
                class_distribution = merged_dataset.count_values("ground_truth.detections.label")  # Calculate class distribution

                print("Class Distribution:", json.dumps(class_distribution, indent=2))

                return merged_dataset, class_distribution

    

    def evaluate_model(self, model: YOLO) -> Dict[str, Any]:
        """
        Evaluate the trained model's performance.
        
        Args:
            model: Trained YOLO model
            
        Returns:
            Dictionary containing evaluation metrics
        """

        print("Evaluating model...")

        results = model.val(data=self.data_yaml_path, device=self.device)  # Evaluate the model
        
        test_dataset = fo.load_dataset(self.dataset_name).match_tags("test")  # Load existing dataset for evaluation
        fps = self._calculate_fps(model, test_dataset)  # Evaluate on test samples to estimate performance

        results_dict = results.results_dict  # Get results dictionary

        final_results = {  # Create a dictionary of evaluation metrics

            "mAP50-95": results_dict.get('metrics/mAP50-95(B)'),  # Mean average precision at IoU thresholds from 0.5 to 0.95
            "mAP50": results_dict.get("metrics/mAP50(B)"),  # Mean average precision at IoU threshold of 0.5    
            "precision": results_dict.get("metrics/precision(B)"),  # Precision
            "recall": results_dict.get("metrics/recall(B)"),  # Recall
            "fps": fps,  # Frames per second

            "inference_device": str(self.device),  # Inference device (GPU or CPU)
            "class_metrics": {  # Class-specific metrics
                cls: {
                    "precision": results_dict.get(f"metrics/{cls}_precision"),  
                    "recall": results_dict.get(f"metrics/{cls}_recall"),  
                    "mAP50": results_dict.get(f"metrics/{cls}_mAP50")  
                }
                for cls in self.classes
            }
        }

        results_path = self.directories["results"] / "evaluation_results.json"
        with open(results_path, "w") as f:

            json.dump(final_results, f, indent=4)  # Ensure indentation

        return final_results

    def _calculate_fps(self, model: YOLO, dataset: fo.Dataset) -> float:

        """
        Calculate frames per second on test dataset.
        
        Args:
            model: Trained YOLO model
            dataset: Test dataset
            
        Returns:
            Average FPS
        """
        print("Calculating FPS...")
        times = []



        with torch.no_grad():  # Disable gradient calculations for faster inference



            for sample in tqdm(dataset, desc="FPS Test"):  # Iterate through the test dataset

                img = cv2.imread(sample.filepath)
                if img is None:

                    raise ValueError("Image cannot be loaded") # Or appropriate error handling
                start_time = time.time()

                _ = model.predict(source=[img], device=self.device)  # Perform inference



                end_time = time.time()
                times.append(end_time - start_time)
        return 1.0 / np.mean(times) if times else 0.0  # Proper averaging logic for timings
        



    # def _process_frame(self, model: YOLO, frame: np.ndarray) -> np.ndarray:
    #     """Process a single frame with GPU acceleration."""
    #     img = self._preprocess_frame(frame)
    #     with torch.inference_mode():
    #         results = model(img, conf=self.conf_thres, iou=self.iou_thres)[0]
    #     self.fps_counter.update()
    #     results_np = results.cpu().numpy()
    #     labels = [self.classes[int(cls)] for *xyxy, conf, cls in results_np.boxes.data]  # Get all predicted labels
        
    #     results = results.plot()
    #     return cv2.cvtColor(results, cv2.COLOR_RGB2BGR)

    def _filter_detections(
        self,
        dataset: fo.Dataset,
        cls: str,
        confidence_thresh: float,
        area_thresh: float
    ) -> None:
        """Filter detections based on confidence and area thresholds."""

        # Check if "ground_truth" field exists
        if "ground_truth" not in dataset.get_field_schema():
            print(f"Warning: 'ground_truth' field not found in dataset. Skipping filtering for class '{cls}'.")
            return

        # Check if "detections" field exists within "ground_truth"
        if "detections" not in dataset.get_field_schema(field="ground_truth"):
            print(f"Warning: 'detections' field not found within 'ground_truth' for class '{cls}'. Skipping filtering.")
            return

        # If both fields exist, proceed with filtering
        dataset.filter_labels(
            "ground_truth",
            (F("confidence") > confidence_thresh) &
            (F("bounding_box")[2] * F("bounding_box")[3] >= area_thresh),
        )






    @staticmethod
    def _convert_classifications_to_detections(dataset: fo.Dataset) -> None:
        """Convert classification labels to detection format."""
        
class FPSCounter:
    """Utility class for FPS calculation."""
    def __init__(self, avg_frames: int = 30):
        self.avg_frames = avg_frames  # Number of frames to average over
        self.times = []  # Store timestamps
        self.fps = 0  # Calculated FPS

    def update(self) -> None:
        """Update internal timers and recalculate FPS."""
        self.times.append(time.time())  # Add current timestamp



        if len(self.times) > self.avg_frames:
            self.times.pop(0)  # Remove oldest timestamp if exceeding average frames

        # if self.times:
        #     self.fps = len(self.times) / (self.times[-1] - self.times[0])  # Correct timeframe if self.times is not empty

class StreamProcessor:
    """Handle real-time video processing with optimized GPU usage."""
    def __init__(
        self,
        model: YOLO,
        device: torch.device,
        img_size: int,
        conf_thres: float,
        iou_thres: float

    ):
        """
        Initialize the StreamProcessor.

        Args:
            model: Trained YOLO model.
            device: Device (GPU or CPU) for inference.
            img_size: Input image size.
            conf_thres: Confidence threshold for detection.
            iou_thres: Intersection over Union threshold.
        """
        self.model = model.to(device)  # Send to device immediately on init
        self.device = device  # Ensure device configuration for GPU use
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.fps_counter = FPSCounter()  # Initialize FPS counter

    

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame with GPU acceleration."""
        img = self._preprocess_frame(frame)  # Preprocess the frame
        with torch.inference_mode():  # Disable gradient calculations
            results = self.model(img, conf=self.conf_thres, iou=self.iou_thres)[0]  # Perform inference
        self.fps_counter.update()  # Update FPS counter
        results = results.plot()  # Plot bounding boxes on the frame
        return cv2.cvtColor(results, cv2.COLOR_RGB2BGR)  # Convert to BGR for display

    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess frame for model input."""
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
        img = cv2.resize(img, (self.img_size, self.img_size))  # Resize to model input size
        img = torch.from_numpy(img).to(self.device).float().div(255)  # Normalize and send to device
        img = img.permute(2, 0, 1)  # Change channel order
        img = img.unsqueeze(0)  # Add batch dimension
        return img




def main():

    """Main execution function with improved error handling."""
    config = ModelConfig(
        img_size=640,
        batch_size=16,
        epochs=3,
        conf_thres=0.5,
        iou_thres=0.45
    )



    system = DetectionSystem(  # Initialize the detection system
        dataset_name="open-images-v7-dataset",
        classes=["Cat", "Dog", "Bird", "Man", "Woman","Human face"],
        config=config
    )

    dataset, class_dist = system.prepare_dataset()  # Prepare the dataset
    model = system.train_model()  # Train the model
    results = system.evaluate_model(model)  # Evaluate the model



    processor = StreamProcessor(  # Initialize the stream processor for real-time detection
        model=model,
        device=system.device,
        img_size=config.img_size,
        conf_thres=config.conf_thres,
        iou_thres=config.iou_thres
    )

    
    cap = cv2.VideoCapture(0)  # Open the default camera

    while cap.isOpened():
        ret, frame = cap.read()  # Read a frame
        if not ret:
            break

        processed_frame = processor.process_frame(frame)  # Process the frame

        cv2.imshow("Detection press""q"" to quit", processed_frame)  # Display the processed frame
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()