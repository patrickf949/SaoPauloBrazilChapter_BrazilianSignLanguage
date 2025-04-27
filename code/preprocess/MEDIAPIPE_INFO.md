## MediaPipe Holistic Model: Output Format & Details

### 🔹 `pose_landmarks`
- **Count**: 33 landmarks
- **Each landmark includes**:
  - `x`, `y`, `z`: normalized coordinates (relative to image width/height)
  - `visibility`: float [0.0, 1.0], confidence the point is visible and unoccluded
- **Rules**:
  - `pose_landmarks` is `None` if no pose is detected in the frame
  - If present, all 33 landmarks are always provided
  - `visibility` is useful to filter out unreliable points (e.g., low confidence if occluded)

---

### 🔹 `face_landmarks`
- **Count**: 468 landmarks
- **Each landmark includes**:
  - `x`, `y`, `z`: normalized coordinates
- **Rules**:
  - `face_landmarks` is `None` if no face is detected
  - If present, all 468 landmarks are always included
  - No `visibility` or `presence` scores — landmarks may be inaccurate if occluded/off-screen

---

### 🔹 `left_hand_landmarks` / `right_hand_landmarks`
- **Count**: 21 landmarks per hand
- **Each landmark includes**:
  - `x`, `y`, `z`: normalized coordinates
- **Rules**:
  - Entire hand group is `None` if the hand is not detected
  - If detected, all 21 landmarks are included regardless of visibility
  - No `visibility` or confidence scores per landmark

---

### 📝 Notes
- All coordinates (`x`, `y`) are **relative to the image dimensions** (from 0 to 1)
- The `z` value is depth — it's **relative**, not in real-world units
- Use `visibility` only available in pose landmarks to filter out low-confidence points
- Landmarks are **not individually None** — the entire group is either present or `None`