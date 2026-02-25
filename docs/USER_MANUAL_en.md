# SAM 3 Labeling Tool - Complete User Manual

## For Beginners

Welcome to the SAM 3 Labeling Tool! This software helps you mark objects in images.

**What is "labeling"?**
Imagine using a highlighter to circle important things in a photo and writing what they are. That's labeling! Computers need these labels to learn how to recognize objects.

**What can this tool do?**
- Select or click on objects in images
- Automatically identify object boundaries (using AI)
- Save annotation data for computer learning

---

## Chapter 1: Understanding the Interface

After starting the program, your browser will open automatically. You'll see:

```
┌─────────────────────────────────────────────────────────────────────┐
│  SAM3    [Images Folder]    [Output Folder]    [Load] [Jump]        │  ← Navigation
├────────┬───────────────────────────────────────┬────────────────────┤
│        │                                       │                    │
│ Tool   │                                       │    Control Panel   │
│ Panel  │         Image Display Area            │                    │
│        │                                       │    - Text Segment  │
│ ○Select│    ←  [    Image    ]  →              │    - Class Select  │
│ ○Click │                                       │    - Output Format │
│ ○Box   │                                       │    - Label Manage  │
│        │                                       │                    │
├────────┴───────────────────────────────────────┴────────────────────┤
│  Status Bar: Shows current progress and operation messages          │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.1 Navigation Bar (Top)

| Component | Description | How to Use |
|-----------|-------------|------------|
| **SAM3** | Program name | Just a title, no need to click |
| **Images Folder** | Tell program where your images are | Enter folder path, e.g., `D:\my_images` |
| **Output Folder** | Where to save annotation results | Enter folder path, e.g., `D:\labels` |
| **Load** | Start reading images | Click after setting paths |
| **Jump** | Jump to specific image number | Enter number, e.g., `50`, then press Enter |

### 1.2 Tool Panel (Left)

Three tools available, only one active at a time:

```
┌──────────────┐
│  Tools       │
├──────────────┤
│ ● Select Labels │  ← Select existing labels
│ ○ Click Segment │  ← Click to auto-segment
│ ○ Box Segment   │  ← Draw box to segment
└──────────────┘
```

**Select Labels** - Use when you want to modify or delete existing labels
**Click Segment** - Easiest! Just click object center, AI will auto-segment
**Box Segment** - Draw a box around object, AI will identify boundaries

---

## Chapter 2: Getting Started (Quick Guide)

### Step 1: Prepare Images

First, put images to label in a folder.

**Recommendation:**
1. Create a folder, e.g., `D:\images_to_label`
2. Copy all images there
3. Image formats: `.jpg`, `.png`, `.jpeg`, `.bmp`

### Step 2: Launch Program

Open Command Prompt (Press Win+R, type `cmd`, press Enter)

Enter:
```
cd sam3_labeler_package
python sam3_labeler.py
```

Wait a few seconds, browser will open automatically.

**If it doesn't open automatically:** Open browser manually, enter `http://localhost:7860`

### Step 3: Load Images

1. In "Images Folder" field, enter your images path
   ```
   D:\images_to_label
   ```

2. In "Output Folder" field, enter where to save results
   ```
   D:\my_labels
   ```

3. Click "Load" button

4. If successful, you'll see the first image!

### Step 4: Start Labeling

Easiest method - "Click Segment":

1. Select "Click Segment" in left tool panel
2. Confirm "Current Class" on right is what you want
3. Click on center of object in image
4. AI will automatically identify and outline object!

**Colored box appearing means success!**

### Step 5: Next Image

Click → arrow on right side of image, or press → arrow key

**Important: Labels auto-save when switching images!**

---

## Chapter 3: Three Labeling Methods Explained

### Method 1: Click Segment (Easiest)

**Best for:** Obvious objects with clear boundaries

**Steps:**
1. Select "Click Segment" tool
2. Select correct class
3. Click center of object
4. Done!

**Tip:** Click as close to center of object as possible

### Method 2: Box Segment (More Precise)

**Best for:** Overlapping objects, when AI makes mistakes

**Steps:**
1. Select "Box Segment" tool
2. Select correct class
3. Click top-left corner of object → screen shows marker
4. Click bottom-right corner of object → AI starts identifying

**Tip:** Box should contain entire object, stay close to edges

### Method 3: Text Segment (Auto-find Objects)

**Best for:** Finding all specific objects at once

**Steps:**
1. In "Text Prompt" field, enter object name
   - Single object: `bottle`
   - Multiple: `bottle, debris, buoy` (comma separated)
2. Click "Segment by Text"
3. AI will find all matching objects!

**Note:** Text entered automatically becomes a new class

---

## Chapter 4: Editing Existing Labels

### 4.1 Selecting Labels

1. Switch to "Select Labels" tool
2. Click label box on image
3. Selected labels show yellow thick border

**Multi-select:** Click each one to add to selection

### 4.2 Deleting Labels

1. Select labels to delete (can multi-select)
2. Click "Delete Selected" button on right
3. Labels disappear!

### 4.3 Changing Class

If you accidentally labeled a "bottle" as "debris":

1. Select that label
2. Choose correct class from dropdown on right
3. Click "Apply" button
4. Done!

---

## Chapter 5: Auto-Save Feature

Good news! This program auto-saves your labels:
- When switching to next image → auto-saves
- When switching to previous image → auto-saves

**You don't need to manually save!**

---

## Chapter 6: Output Files Explained

Program generates these files:

```
D:\my_labels\
├── images\              ← Labeled image copies
│   ├── image001.jpg
│   └── image002.jpg
│
├── labels\              ← OBB format labels
│   ├── image001.txt
│   └── image002.txt
│
├── labels_seg\          ← Polygon format labels
│   ├── image001.txt
│   └── image002.txt
│
├── masks\               ← Mask images (if enabled)
│   ├── image001.png
│   └── image002.png
│
└── classes.txt          ← All class names
```

| Format | Location | Purpose |
|--------|----------|---------|
| OBB | labels/*.txt | For YOLO OBB model training (rotated boxes) |
| Seg | labels_seg/*.txt | For YOLO-Seg model training (instance segmentation) |
| Mask | masks/*.png | For semantic segmentation or mask creation |

**Beginner tip:** Keep default OBB and Seg checked!

---

## Chapter 7: Frequently Asked Questions

### Q1: Clicked object but nothing happened?

**Possible causes:**
- Wrong tool selected → Confirm "Click Segment" is selected
- AI can't recognize → Try "Box Segment" instead

**Solution:**
1. Check "Use Box if SAM Fails" on right
2. This creates box annotation even if AI fails

### Q2: Label shape is wrong?

**Solution:**
1. Delete wrong label
2. Use "Box Segment" for more precise selection
3. Or try clicking different positions

### Q3: Forgot to select class before labeling?

**Solution:**
1. Select that label
2. Choose correct class
3. Click "Apply"

### Q4: Program crashed, are labels lost?

**Don't worry!**
- As long as you switched images before, previous labels are saved
- Restart program, load same folder to continue

---

## Chapter 8: Good Labeling Habits

### 1. Label Completely
Make sure entire object is inside the box

### 2. Label Tightly
Keep box close to object, don't leave too much space

### 3. One Object Per Label
Label each object separately, don't put multiple objects in one label

### 4. Be Consistent
Same objects should always use same class name

### 5. Backup Regularly
Every 50-100 images, backup your output folder

---

## Chapter 9: Keyboard Shortcuts

| Key | Function |
|-----|----------|
| ← (Left Arrow) | Previous image |
| → (Right Arrow) | Next image |
| Delete | Delete selected labels |
| Escape | Cancel selection |

---

## Glossary

| Term | Explanation |
|------|-------------|
| **Annotation/Labeling** | Marking object location and class on images |
| **Class** | Type of object, e.g., "bottle", "boat" |
| **OBB** | Oriented Bounding Box, rotated rectangle |
| **Seg** | Segmentation, precise object boundary |
| **Mask** | Black and white image showing object position |
| **SAM** | Segment Anything Model, AI segmentation by Meta |
| **AI** | Artificial Intelligence |
| **GPU** | Graphics card, accelerates AI computing |

---

Happy labeling!
