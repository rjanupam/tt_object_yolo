import cv2
import numpy as np
import supervision as sv
import torch
import subprocess
from ultralytics import YOLO

def download_model(model_name="yolov8x.pt"):
    try:
        model = YOLO(model_name)
    except Exception as e:
        print(f"Downloading {model_name}...")
        torch.hub.download_url_to_file(f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{model_name}", model_name)
        model = YOLO(model_name)
    return model

model = download_model()

def analyze_room_occupancy(image_path):
    image = cv2.imread(image_path)
    results = model(image)[0]
    detections = sv.Detections(
        xyxy=results.boxes.xyxy.cpu().numpy(),
        confidence=results.boxes.conf.cpu().numpy(),
        class_id=results.boxes.cls.cpu().numpy().astype(int)
    )
    people_count = sum(1 for class_id in detections.class_id if model.names[class_id] == "person")
    chair_count = sum(1 for class_id in detections.class_id if model.names[class_id] == "chair")
    bench_count = sum(1 for class_id in detections.class_id if model.names[class_id] == "bench")
    room_area = image.shape[0] * image.shape[1]
    occupied_area = sum((box[2] - box[0]) * (box[3] - box[1]) for box in detections.xyxy)
    available_space = (room_area - occupied_area) / room_area
    chair_capacity = chair_count
    bench_capacity = bench_count * 2
    total_seating = chair_capacity + bench_capacity
    capacity_factor = 0.7
    estimated_capacity = int((room_area * capacity_factor) / (room_area / (people_count + total_seating)))
    analysis = {
        "people_count": people_count,
        "chair_count": chair_count,
        "bench_count": bench_count,
        "total_seating": total_seating,
        "estimated_capacity": estimated_capacity,
        "available_space": available_space
    }
    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    annotated_image = bounding_box_annotator.annotate(scene=image.copy(), detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image,
        detections=detections,
        labels=[f"{model.names[class_id]}: {detections.confidence[i]:.2f}" 
                for i, class_id in enumerate(detections.class_id)]
    )
    return annotated_image, analysis

def interpret_results(analysis):
    people_count = analysis["people_count"]
    total_seating = analysis["total_seating"]
    estimated_capacity = analysis["estimated_capacity"]
    available_space = analysis["available_space"]
    bench_count = analysis["bench_count"]
    interpretation = []
    if people_count == 0:
        interpretation.append("The room is currently empty.")
    elif people_count < estimated_capacity:
        interpretation.append(f"The room contains {people_count} people and has space for more.")
    else:
        interpretation.append(f"The room is at or near capacity with {people_count} people.")
    interpretation.append(f"There are {total_seating} total seats available (including {bench_count} benches).")
    if available_space > 0.3:
        interpretation.append("There is significant available space in the room.")
        if total_seating < estimated_capacity:
            interpretation.append("More seating could be added to accommodate more people.")
    elif available_space > 0.1:
        interpretation.append("There is some available space in the room.")
        if total_seating < people_count:
            interpretation.append("A few more seats could be added if needed.")
    else:
        interpretation.append("The room appears to be quite full.")
        if total_seating > people_count:
            interpretation.append("There are more seats than people, suggesting maximum utilization.")
    return "\n".join(interpretation)

if __name__ == "__main__":
    while True:
        # Prompt the user for an image path
        image_path = input("Enter the path to the image file (or type 'exit' to quit): ").strip()

        # Check if the user wants to exit the program
        if image_path.lower() == "exit":
            print("Exiting the program.")
            break
        
        try:
            # Analyze the room occupancy
            annotated_image, analysis = analyze_room_occupancy(image_path)
            
            # Display the analysis results
            print("Room Analysis:")
            print(f"People count: {analysis['people_count']}")
            print(f"Chair count: {analysis['chair_count']}")
            print(f"Bench count: {analysis['bench_count']}")
            print(f"Total seating: {analysis['total_seating']}")
            print(f"Estimated capacity: {analysis['estimated_capacity']}")
            print(f"Available space: {analysis['available_space']:.2%}")
            print("\nInterpretation:")
            print(interpret_results(analysis))

            # Save and open the annotated image
            output_image_path = "annotated_room.jpg"
            cv2.imwrite(output_image_path, annotated_image)
            subprocess.run(["eog", output_image_path])

        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please try again with a valid image path.")
