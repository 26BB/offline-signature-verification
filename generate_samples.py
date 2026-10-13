import os
import numpy as np
import cv2

def create_sample_signatures(output_dir="dataset/raw"):
    os.makedirs(output_dir, exist_ok=True)
    
    students = ["student_001", "student_002"]
    
    for s in students:
        s_dir = os.path.join(output_dir, s)
        f_dir = os.path.join(s_dir, "forged")
        os.makedirs(s_dir, exist_ok=True)
        os.makedirs(f_dir, exist_ok=True)
        
        # Generate 3 dummy genuine signatures with small random variations
        for i in range(1, 4):
            canvas = np.ones((200, 400), dtype=np.uint8) * 255
            # Draw cursive-like curves
            offset_y = np.random.randint(-5, 6)
            offset_x = np.random.randint(-5, 6)
            
            pts = np.array([
                [50 + offset_x, 100 + offset_y],
                [100 + offset_x, 60 + offset_y],
                [150 + offset_x, 140 + offset_y],
                [220 + offset_x, 80 + offset_y],
                [300 + offset_x, 120 + offset_y],
                [350 + offset_x, 70 + offset_y]
            ], np.int32)
            
            thickness = np.random.choice([2, 3])
            cv2.polylines(canvas, [pts], False, (0,), thickness=thickness, lineType=cv2.LINE_AA)
            cv2.imwrite(os.path.join(s_dir, f"g_{i:02d}.png"), canvas)
            
        # Generate 1 forged signature
        canvas_f = np.ones((200, 400), dtype=np.uint8) * 255
        pts_f = np.array([
            [50, 140],
            [120, 130],
            [180, 50],
            [260, 160],
            [340, 80]
        ], np.int32)
        cv2.polylines(canvas_f, [pts_f], False, (0,), thickness=2, lineType=cv2.LINE_AA)
        cv2.imwrite(os.path.join(f_dir, "f_01.png"), canvas_f)

    print("Sample signatures created successfully in dataset/raw!")

if __name__ == "__main__":
    create_sample_signatures()
