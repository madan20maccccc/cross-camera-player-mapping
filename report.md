# 📝 Report – Cross-Camera Player Mapping

## 🎯 Objective

The goal of this project is to perform **cross-camera player mapping** between two synchronized videos (broadcast and tacticam views) of a football match. The task involves matching each detected player in the tacticam view to the corresponding player in the broadcast view, using consistent identity labels.

This work is submitted for **Option 1** of a computer vision assignment.
## 🧠 Methodology Summary

The approach to cross-camera player mapping involved the following key stages:

### 1. Player Detection
A fine-tuned **YOLOv11** object detection model (`best.pt`) was used to detect players in each frame of both input videos (`broadcast.mp4` and `tacticam.mp4`). YOLOv11 provides accurate and real-time detection capabilities suited for detecting multiple players in complex scenes. For every detected player, bounding box coordinates were extracted and the corresponding player region was cropped and saved as an image. This cropping allowed focused analysis on individual players without background noise.

### 2. Visual Feature Extraction
To characterize players visually beyond bounding boxes, the **OpenAI CLIP (ViT-B/32)** model was employed. CLIP, a vision-language model, can generate high-quality image embeddings capturing semantic features such as jersey color, posture, and appearance details. Each cropped player image was preprocessed and passed through the CLIP encoder to extract a fixed-length vector embedding. These embeddings serve as visual fingerprints to compare players across camera views.

### 3. Spatial Normalization
Since the two cameras capture different perspectives and resolutions, direct spatial comparison is not straightforward. To mitigate this, the center point `(x, y)` of each player’s bounding box was computed and then normalized by dividing by the frame resolution `(1280x720)`. This normalization scales coordinates to a [0, 1] range, making spatial distances between players comparable across videos despite differences in camera angle or zoom.

### 4. Player Matching
The final player matching is based on a combined similarity score defined as:

final_score = visual_similarity - α * spatial_distance

- **Visual Similarity:** Measured as the cosine similarity between CLIP embeddings of tacticam and broadcast players. This quantifies how visually alike the players are, with values ranging from 0 (no similarity) to 1 (identical).
  
- **Spatial Distance:** Calculated as the Euclidean distance between the normalized center coordinates of the players. This captures the relative spatial proximity, penalizing matches that are visually similar but spatially far apart.
  
- **α (alpha):** A weighting factor balancing the importance of spatial distance in the final score. After experimentation, a value of 0.5 was chosen, meaning spatial proximity reduces the score by half the distance value.

For each player detected in the tacticam video, the broadcast player with the highest `final_score` was selected as the corresponding match. This approach prioritizes both visual similarity and plausible spatial correspondence to improve mapping accuracy.

### 5. Output Generation
The mapping results were saved in `player_mapping.csv` with three columns:
- `frame`: Frame number in the tacticam video
- `player_idx`: Index of the player detected in that frame
- `matched_broadcast_player`: Index of the matched player in the broadcast video

This output provides a frame-wise identity mapping of players between the two camera views, enabling consistent tracking of players across feeds.

---

This methodology balances interpretability, computational efficiency, and practical constraints by avoiding complex temporal tracking or re-identification models, making it suitable for the assignment’s requirements.
## ⚠️ Challenges & Limitations

During the development of this player mapping solution, several challenges and limitations were encountered:

### 1. Visual Ambiguity
Players often wear similar jerseys, and the camera angles can cause visual distortion or occlusion. This sometimes results in ambiguous CLIP embeddings, making it hard to distinguish between players based solely on appearance.

### 2. Spatial Discrepancies
Due to different camera perspectives (wide broadcast vs. close tacticam), spatial coordinates can be misleading. Normalization helps but cannot fully compensate for depth differences, camera angles, or player movement in 3D space.

### 3. Frame-wise Matching Without Temporal Context
The matching is done frame-by-frame independently without leveraging temporal continuity or tracking. This can cause inconsistent identity assignments across frames for the same player.

### 4. No Re-identification or Tracking Models Used
State-of-the-art re-identification or tracking methods (e.g., DeepSORT) were not employed, as per the assignment scope. While simplifying the pipeline, this reduces robustness in complex scenes or occlusions.

### 5. Computational Load
Extracting CLIP embeddings for all cropped images is computationally expensive and time-consuming, especially for longer videos or high frame rates.

### 6. Parameter Sensitivity
The alpha (`α`) parameter controlling spatial penalty was heuristically set to 0.5. Different videos or scenarios may require tuning for optimal performance.

---

### Future Improvements
- Incorporate temporal tracking or re-identification models to maintain consistent identities across frames.
- Use 3D spatial reasoning or homography to better relate player positions across camera views.
- Optimize embedding extraction for speed, e.g., batch processing or using lighter models.
- Experiment with adaptive weighting between visual and spatial similarity based on scene context.

---

Despite these challenges, the implemented method provides a reasonable and interpretable solution for cross-camera player mapping using only visual appearance and spatial cues.
## 🚀 Submission Instructions & How to Run

### Submission Contents
Ensure your submission folder/repository contains:
- `cross_camera_player_mapping.ipynb` — The full Google Colab notebook with all code and comments
- `player_mapping.csv` — The final CSV output with player mappings
- `README.md` — This documentation file explaining the project and setup
- `report.md` (or `report.pdf`) — This detailed project report explaining your approach, challenges, and results
- `best.pt` — YOLOv11 model weights file (either included or accessible via your Drive link)

---

### How to Run the Notebook

1. **Upload files to Google Drive:**  
   Create a folder (e.g., `/MyDrive/int1/`) and upload the two input videos (`broadcast.mp4` and `tacticam.mp4`) and the model weights file (`best.pt`).

2. **Open the Colab notebook:**  
   Open `cross_camera_player_mapping.ipynb` in Google Colab.

3. **Mount Google Drive:**  
   Run the provided cell to mount your drive:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')

4. Set file paths:
Verify that file paths in the notebook point to your Drive folder.

5. Run the notebook step-by-step:
   Detect players in both videos
   Crop detected players and save images
   Extract CLIP embeddings from cropped images
   Normalize player positions
   Calculate combined similarity scores and match players
   Save final mappings to player_mapping.csv

6. Download/Export Results:
The final mapping CSV is saved at /content/player_mapping.csv. Download or export this file for submission or further analysis.

---

### Notes
- The notebook is modular and well-commented to facilitate understanding and reproducibility.
- The approach performs **frame-wise matching only**, without temporal tracking or re-identification models.
- Ensure all required dependencies (such as `ultralytics`, `clip`, `opencv-python`, `pandas`, and others) are installed as described in the notebook.
- The method emphasizes simplicity and interpretability over complex tracking algorithms, suitable for the assignment scope.
- For longer videos or higher frame rates, consider optimizing embedding extraction to reduce computational load.

---

### Contact
For any questions or support regarding this project, please reach out to:

**Madan**  
B.Tech Artificial Intelligence & Data Science  
Amrita Vishwa Vidyapeetham  
Email: madan.m200607@gmail.com




