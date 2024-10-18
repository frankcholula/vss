import numpy as np

class Extractor:
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
    def extract_globalRGBhisto(img, bins=32, encoding=False) -> np.ndarray:
        """
        Extracts a global RGB histogram from the input image.

        This method computes the histogram for each of the B, G, and R channels of the input image.
        The histogram is computed with a specified number of bins and is normalized to sum to 1.

        Parameters:
        img (numpy.ndarray): The input image, assumed to be normalized to the range [0, 1].
        bins (int): The number of bins to use for the histogram. Default is 32.
        encoding (bool): A flag for future use, currently not implemented. Default is False.

        Returns:
        numpy.ndarray: A flattened and normalized histogram of the RGB channels.
        """
        hist = np.zeros((3, bins))
        for i in range(3):
            # compute the histogram for each channel, but range is [0, 1] because we already normalized the image
            hist[i] = np.histogram(img[:, :, i], bins=bins, range=(0, 1))[0]
        # flatten and normalize the histogram
        hist_flat = hist.flatten()
        hist_normalized = hist_flat / np.sum(hist_flat)
        return hist_normalized

    @staticmethod
    def extract_globalRGBencoding(img, base) -> np.ndarray:
        """
        Encodes the RGB channels of the input image into a single channel using a polynomial representation.

        This method combines the R, G, and B channels of the input image into a single channel by treating
        each pixel as a polynomial with the specified base. The resulting encoded image is then flattened.

        Parameters:
        img (numpy.ndarray): The input image, assumed to be normalized to the range [0, 1].
        base (int): The base to use for the polynomial representation.

        Returns:
        numpy.ndarray: A flattened array representing the encoded image.
        """
        R = img[:, :, 0]
        G = img[:, :, 1]
        B = img[:, :, 2]
        poly_repr = R * base ** 2 + G * base + B
        return poly_repr.flatten()
