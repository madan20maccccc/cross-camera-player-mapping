# 🎯 Cross-Camera Player Mapping

This project performs player mapping between two different video views (broadcast and tacticam) of the same football match. The objective is to match detected players across the two camera feeds, ensuring consistent identity mapping using a combination of object detection, visual embeddings, and spatial analysis.

---

## 📁 Repository Contents

| File/Folder                      | Description                                                  |
|----------------------------------|--------------------------------------------------------------|
| `cross_camera_player_mapping.ipynb` | Main Google Colab notebook with all code                     |
| `player_mapping.csv`             | Final output mapping between players across cameras          |
| `best.pt`                        | YOLOv11 object detection model (not included here; in Drive) |
| `README.md`                      | This instruction file                                        |
| `report.md`                      | Project explanation and methodology                          |

---

## 🧠 Project Methodology

### 1. 🎥 Input Videos
- `broadcast.mp4` — wide-angle view  
- `tacticam.mp4` — tactical camera view  
*(Both videos show the same match and are aligned in time)*

### 2. 🧍 Player Detection
- Used a fine-tuned **YOLOv11** model (`best.pt`) to detect players in both videos  
- Cropped each detected player from the frame and saved as individual images

### 3. 🎨 Visual Feature Extraction
- Used **CLIP (ViT-B/32)** model from OpenAI to extract embeddings for each cropped player image  
- These embeddings capture how each player **looks visually** (jersey, posture, etc.)

### 4. 📍 Spatial Normalization
- Computed the **center coordinates** `(x, y)` of each player bounding box  
- Normalized coordinates by frame resolution `(1280x720)` for scale fairness

---

## 🔗 5. Player Matching Logic

To identify which broadcast player corresponds to a tacticam player, we use this scoring formula:

final_score = visual_similarity - α * spatial_distance

Where:
- `visual_similarity`: Cosine similarity between image embeddings (range: 0 to 1)  
- `spatial_distance`: Euclidean distance between player positions  
- `α` (alpha): Weight to control the spatial penalty (we used 0.5)

✅ This means:
- Higher **visual similarity** = better match  
- Lower **spatial distance** = better match  

We subtract the spatial penalty so that visually similar and spatially close players are favored.

🔄 The player in the broadcast feed with the highest `final_score` is selected as the match.

---

## 🛠️ How to Run This Project (in Google Colab)

### Step 1: Upload Required Files to Drive
Create a folder in your Google Drive (e.g., `/MyDrive/int1/`) and upload:
- `broadcast.mp4`
- `tacticam.mp4`
- `best.pt` (YOLOv11 model weights)

### Step 2: Open Colab and Run the Notebook
- Open `cross_camera_player_mapping.ipynb` in Google Colab  
- Mount your Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Then follow each code cell in the notebook to:

- Detect players  
- Crop images  
- Extract CLIP embeddings  
- Normalize positions  
- Match tacticam players to broadcast players  
- Save output

### Step 3: Output
Final mapping will be saved to:  
`/content/player_mapping.csv`  

You can export this to your Drive or GitHub for submission.

📄 **Output Format**  
The output file `player_mapping.csv` contains:

| Column                  | Description                                 |
|-------------------------|---------------------------------------------|
| `frame`                 | Frame number from tacticam video            |
| `player_idx`            | Detected player index in that frame         |
| `matched_broadcast_player` | Closest matching player from broadcast view |

📌 **Notes**
- This project uses frame-wise matching only (no temporal tracking)  
- DeepSORT or re-identification is not used — this is acceptable for Option 1 of the assignment  
- CLIP embeddings and spatial normalization provide a strong match based on appearance + location  
- Matching is qualitative (you can visualize the correctness using images)

---

## 🔗 External Resources: Model & Input Videos

The files required to run this project — including the YOLOv11 model weights (`best.pt`) and the input videos (`broadcast.mp4` and `tacticam.mp4`) — are too large to be uploaded directly to GitHub.

You can download all these files from the following Google Drive folder:

👉 [Download Model & Videos](https://drive.google.com/drive/folders/12K05YMEXg8jPWaRLKdtQ1yG5C6u6pFsb?usp=drive_link)

### Instructions:

1. Download the entire folder or individual files from the above link.  
2. Place the downloaded files in the working directory or the appropriate path as specified in the notebook.  
3. Then, run the notebook as instructed.

---

## 🙋‍♂️ Author

This solution was developed for **Option 1: Cross-Camera Player Mapping** as part of a computer vision assignment under real-world constraints and open-ended problem-solving.

All work was performed using **Python**, **Google Colab**, and **open-source models**, including:
- **YOLOv11** for object detection  
- **OpenAI CLIP (ViT-B/32)** for visual feature embeddings  
- **NumPy, pandas, scikit-learn** for data handling and similarity scoring  
- **OpenCV** for image processing and image cropping

Developed by **Madan. M**, B.Tech Artificial Intelligence & Data Science,  
**Amrita Vishwa Vidyapeetham**.

> The project combines visual embeddings and spatial reasoning to identify corresponding players across two camera views of the same football match, without relying on tracking or re-identification models. The approach emphasizes interpretability and modularity over complexity.
