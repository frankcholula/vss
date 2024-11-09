# cvpr
Assignments for Computer Vision and Pattern Recogition


## Checklist
- [x] **Req No. 1: Global Colour Histogram** (30%)  
  - Implement the global colour histogram from lecture 7 (slides 4,5) using a Euclidean distance metric.  
  - Experiment with different levels of RGB quantization.

- [x] **Req No. 2: Evaluation Methodology** (25%)  
  - Compute precision-recall (PR) statistics for each of your experiments, e.g., PR for the top 10 results.  
  - Plot the PR curve.  
  - If similarity is defined in terms of object categories, compute a confusion matrix.  
  - Discuss and analyze your results (e.g., which experiments were most successful, which images worked well, and why given your descriptor choice).

- [x] **Req No. 3: Spatial Grid (Colour and Texture)** (15%)  
  - Implement gridding of the image and concatenate features from grid cells to form the image descriptor.  
  - Experiment with colour and/or texture features.  
  - Experiment with different levels of angular quantization for texture features.

- [x] **Req No. 4: Use of PCA** (15%)  
  - Use PCA to project your image descriptors into a lower-dimensional space.  
  - Explore the use of Mahalanobis distance as an alternative distance metric.  
  - Analyze whether performance improves.

- [x] **Req No. 5: Different Descriptors and Distance Measures** (15%)  
  - Experiment with different choices of distance measures (e.g., L1 norm) and note their effect on performance.  
  - Discover and try out other distance measures or descriptors not covered in the module.

- [ ] **Req No. 6: Bag of Visual Words Retrieval (Hard)** (40%)  
  - Implement a basic BoVW system using a sparse feature detector (e.g., Harris or SIFT keypoint detector) and a descriptor (e.g., SIFT descriptor).  
  - Use k-Means to create the codebook.  
  - Compare the performance with other descriptors you have tried.

- [ ] **Req No. 7: Object Classification Using SVM (Hard)** (30%)  
  - Apply an SVM to classify image categories (e.g., “bike” or “sheep”) based on extracted descriptors.  
  - Note: This is classification, not strictly visual search.

- [ ] **Req No. 8: Extra Credit** (20%)  
  - Propose and implement your own idea based on the above themes or an entirely new concept.  
  - Focus on technical merit related to Computer Vision, not UI or fancy coding.
