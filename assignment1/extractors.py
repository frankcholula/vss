import numpy as np

class Extractors:
    @staticmethod
    def extract_random(img) -> np.ndarray:
        """
        Generate a random row vector with 30 elements.
        
        This function returns a row vector [rand rand .... rand] representing 
        an image descriptor computed from the image 'img'.
        
        Note: 'img' is expected to be a normalized RGB image (colors range [0,1] not [0,255]).
        
        Parameters:
        img (numpy.ndarray): The input image.
        
        Returns:
        numpy.ndarray: A random row vector with 30 elements.
        """
        return np.random.rand(1, 30)
    
    @staticmethod
    def extract_rgb(img) -> np.ndarray:
        """
        Compute the average red, green, and blue values as a basic color descriptor.
        
        This function calculates the average values for the blue, green, and red channels
        of the input image and returns them as a feature vector.
        
        Note: OpenCV uses BGR format, so the channels are accessed in the order B, G, R.
        
        Parameters:
        img (numpy.ndarray): The input image.
        
        Returns:
        numpy.ndarray: A feature vector containing the average B, G, and R values.
        """
        B = np.mean(img[:, :, 0])
        G = np.mean(img[:, :, 1])
        R = np.mean(img[:, :, 2])
        return np.array([R, G, B])
    
    @staticmethod
    def extract_globalRGBhisto(img, bins=32) -> np.ndarray:
        hist = np.zeros((3, bins))
        for i in range(3):
            # compute the histogram for each channel, but range is [0, 1] because we already normalized the image
            hist[i] = np.histogram(img[:, :, i], bins=bins, range=(0, 1))[0]
        # flatten and normalize the histogram
        hist_flat = hist.flatten()
        hist_normalized = hist_flat / np.sum(hist_flat)
        return hist_normalized
