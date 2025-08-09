<img width="3188" height="1202" alt="frame (3)" src="https://github.com/user-attachments/assets/517ad8e9-ad22-457d-9538-a9e62d137cd7" />


# PETALPLATE 🎯


## Basic Details
### Team Name: PixelNoise


### Team Members
- Team Lead: Abhinav P G - GEC PALAKKAD
- Member 2: Anupama J A - GEC PALAKKAD

### Project Description
1. Chapati Roundness and Spot Detection
This module uses computer vision techniques to analyze chapati images for two key quality factors: roundness and surface defects. The image is preprocessed, edges are detected, and contours are analyzed to calculate a roundness score. HSV color filtering and morphological operations are applied to detect and highlight surface spots or burns. The system helps in automated quality grading for food inspection.

2. Marigold Petal Counter
This module automatically counts the petals of a marigold flower from an input image. Using HSV color filtering to isolate the yellow-orange hue range, the system extracts the flower’s contour and generates an angular histogram. Peaks in this histogram represent individual petals, which are then counted after smoothing and merging nearby peaks. This can be used in agricultural research, plant health monitoring, and yield studies.

### The Problem (that doesn't exist)
Every day, millions suffer in silence… from slightly oval chapatis and marigolds with petal counts that are probably fine. Dinner tables across the world are plagued by imperfectly round bread, and gardens are filled with flowers whose petal numbers go unverified. How can humanity truly progress if we don’t know whether our chapati is perfectly circular or if our marigold meets the “ideal petal standard”?

### The Solution (that nobody asked for)
We bring you the ultimate fusion of culinary and botanical precision: a computer vision system that checks the roundness of chapatis and counts marigold petals — because why should bread and flowers escape scientific scrutiny? With finely tuned HSV filters, edgy Canny magic, and a petal-detecting angular histogram, we ensure your dinner and your garden both pass our absurdly high standards. It’s quality control… where quality control was never needed.

## Technical Details
### Technologies/Components Used
For Software:
Languages used: Python
Frameworks used:Streamlit 
Libraries used: 
OpenCV – Image processing, contour detection, color filtering (HSV), morphological operations
NumPy – Numerical computations, histogram generation
Matplotlib – Visualization of analysis results and histograms
Streamlit – For building an interactive UI to upload and analyze images
Tools used:
Vs code
Git/GitHub – Version control


# Run
streamlit run filename


### Project Demo
# Video
[demo video of chapathi perfect test.py](https://github.com/Abhinav-P-G/PetalPlate/blob/main/Screencast%20from%202025-08-09%2015-45-39.mp4))
[demo video of petal_counter.py](video_2025-08-09_16-11-56.mp4)


## Team Contributions
- Abhinav P G: Chapati roundness,brown spot,toast level detection
- Anupama J A: Petal counter
