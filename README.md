## Color Customisation of Clothing using Deep Learning  

The proposed architecture for saree image analysis consists of three main steps: human detection and background removal, removal of body and detection of border, and color changing. Here's an overview of how the process works:

### 1. Human Detection and Background Removal:
- Images are classified based on the presence or absence of a human using a Faster R-CNN model.
- If a human is detected, the background is removed using the MODNet model.
- Human parts like the face and hands are masked using a trained Mask R-CNN model.

### 2. Body and Border Detection:
- In cases where no human is present, the body and border of the saree are recognized using separate trained Mask R-CNN models.
- Custom trained Mas R-CNN models are utilized to detect the saree's body and border regions.

### 3. Color Changing:
- The image is converted to the HSV color space.
- Hue histograms are generated for each area of interest. Different components tend to have distinct yet uniform colors.
- Dominant hue values from the histograms are employed to perform color swaps in the HSV color space.
- Saturation and brightness values are maintained.

### System Architecture:
![image](https://github.com/Arya-adesh/Customisation-of-clothing-using-MaskRCNN/assets/84959568/0387b8a6-c8f6-40a9-9422-71ad2f9b111d)
![image](https://github.com/Arya-adesh/Customisation-of-clothing-using-MaskRCNN/assets/84959568/ae063cda-1123-4937-9e11-cef4374f70a8)


the figures illustrates how the application operates. Users can upload saree images through the web application's frontend, built using HTML, CSS, and JavaScript. This frontend serves as an interface for input and result display. The user history includes uploaded and modified saree images, enabling retailers to understand user preferences.
Users have the option to select colors from a palette, enabling personalized color changes for saree components.

