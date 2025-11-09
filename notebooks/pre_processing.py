import numpy as np
import pandas as pd
import zipfile
import os
import cv2 as cv
from sklearn.preprocessing import MinMaxScaler
from skimage.morphology import reconstruction
from skimage.filters import meijering, sobel, threshold_niblack, threshold_sauvola, median

class Segmentation:
    """Applies various image segmentation and filtering techniques."""

    def __init__(self, width=2048, height=2048):
        """Initializes the Segmentation class."""
        self.__width = width
        self.__height = height

    def _reshape_to_2d(self, img):
        """Reshapes flattened image data to 2D."""
        try:
            return img.reshape(self.__height, self.__width)
        except ValueError:
            # Attempt to infer dimensions if default doesn't match
            side = int(np.sqrt(img.shape[0]))
            if side * side == img.shape[0]:
                self.__height, self.__width = side, side
                return img.reshape(self.__height, self.__width)
            else:
                raise ValueError("Could not reshape image to square dimensions.")

    def _reshape_to_1d(self, img):
        """Reshapes 2D image data back to 1D."""
        return img.reshape(self.__height * self.__width,)

    def _apply_to_images(self, func, images, **kwargs):
        """Helper function to apply a method to each image in a list/array."""
        processed_images = []
        original_h, original_w = self.__height, self.__width
        for img in images:
            # Temporarily set dimensions based on current image before processing
            side = int(np.sqrt(img.shape[0]))
            if side * side == img.shape[0]:
                 self.__height, self.__width = side, side
            else:
                 # Keep original if not square - might error in func if not handled
                 self.__height, self.__width = original_h, original_w

            processed_images.append(func(img, **kwargs))
        
        # Restore original default dimensions
        self.__height, self.__width = original_h, original_w
        return np.array(processed_images)


    def transform_color(self, images):
        """Scales image pixel values to 0-255 range using MinMaxScaler."""
        return self._apply_to_images(self._scale, images)

    def transform_size(self, images, w=1024, h=1024):
        """Resizes images to specified width (w) and height (h)."""
        # Note: This changes the internal width/height for subsequent operations
        # on the *returned* data. Consider resetting if needed elsewhere.
        resized_images = self._apply_to_images(self._resize, images, w=w, h=h)
        self.__height = h
        self.__width = w
        return resized_images

    def apply_dilation(self, images):
        """Applies morphological dilation-based reconstruction."""
        return self._apply_to_images(self._dilation, images)

    def apply_filter(self, images, function_name, k=2):
        """Applies a specific skimage filter."""
        return self._apply_to_images(self._filter, images, func_name=function_name, k=k)

    def apply_equalize_hist(self, images):
        """Applies standard histogram equalization."""
        return self._apply_to_images(self._equ_hist_image, images)

    def apply_clahe(self, images, k=8):
        """Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)."""
        return self._apply_to_images(self._clahe_hist_image, images, k=k)

    def apply_HED(self, images):
        """Applies Histogram Equalization then Dilation reconstruction."""
        equalized = self.apply_equalize_hist(images)
        # Dilation needs original dimensions potentially reset if resizing happened
        # Assuming dilation should run on the equalized image size
        return self.apply_dilation(equalized)

    def apply_CHD(self, images, k=2):
        """Applies CLAHE then Dilation reconstruction."""
        clahe_images = self.apply_clahe(images, k=k)
        # Similar dimension consideration as apply_HED
        return self.apply_dilation(clahe_images)

    # --- Private Helper Methods ---

    def _scale(self, img):
        img_2d = self._reshape_to_2d(img)
        scaler = MinMaxScaler(feature_range=(0, 255))
        # Scaler expects 2D data, fit_transform works directly
        img_scaled_2d = scaler.fit_transform(img_2d)
        return self._reshape_to_1d(img_scaled_2d)

    def _resize(self, img, w, h):
        img_2d = self._reshape_to_2d(img)
        # Ensure dtype compatibility if needed, uint16 might need scaling first
        if img_2d.dtype != np.uint8:
             # Basic scaling before resize if needed, consider MinMax scaling instead
             img_2d = cv.normalize(img_2d, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
        img_resized = cv.resize(img_2d, (w, h))
        # No need to reshape back to 1D here, let _apply_to_images handle it if needed
        # Update internal dimensions for the *next* operation on this image batch
        # self.__height = h
        # self.__width = w
        # Reshape to 1D consistent with other methods returning 1D arrays
        return img_resized.reshape(w*h,)


    def _dilation(self, img):
        img_2d = self._reshape_to_2d(img)
        mask = img_2d
        seed = np.copy(img_2d)
        seed[1:-1, 1:-1] = img_2d.min() # Use min for dilation reconstruction base
        rec = reconstruction(seed, mask, method='dilation')
        # The background is `rec`, subtract it from original
        img_processed = img_2d - rec
        return self._reshape_to_1d(img_processed)

    def _filter(self, img, func_name, k=2):
        img_2d = self._reshape_to_2d(img).astype(np.float64) # Filters often prefer float
        
        # Convert function name string to actual function call
        filter_function = None
        if func_name.lower() == "sobel":
            filter_function = sobel
        elif func_name.lower() == "meijering":
            filter_function = lambda i: meijering(i, sigmas=[k])
        elif func_name.lower() == "median":
            # Median might need uint8 input depending on skimage version
             img_2d_uint8 = cv.normalize(img_2d, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
             img_filtered = median(img_2d_uint8)
             return self._reshape_to_1d(img_filtered) # Return early for median
        elif func_name.lower() == "niblack":
            filter_function = lambda i: threshold_niblack(i, k=k)
        elif func_name.lower() == "sauvola":
            filter_function = lambda i: threshold_sauvola(i, k=k)
        else:
            raise ValueError(f"Unknown filter function: {func_name}")

        img_filtered = filter_function(img_2d)
        return self._reshape_to_1d(img_filtered)

    def _equ_hist_image(self, img):
        # Convert to uint8 for equalizeHist
        img_2d_uint8 = cv.normalize(self._reshape_to_2d(img), None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
        equ = cv.equalizeHist(img_2d_uint8)
        return self._reshape_to_1d(equ)

    def _clahe_hist_image(self, img, k):
         # Convert to uint8 for CLAHE
        img_2d_uint8 = cv.normalize(self._reshape_to_2d(img), None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
        clahe = cv.createCLAHE(clipLimit=float(k)) # clipLimit expects float
        clahe_img = clahe.apply(img_2d_uint8)
        return self._reshape_to_1d(clahe_img)


class Dataset:
    """Handles loading and structuring of the JSRT dataset."""

    def __init__(self,
                 nodule_zip="Data/Nodule154images.zip",
                 non_nodule_zip="Data/NonNodule93images.zip",
                 nodule_bmp_folder="Data/bmp/nodule",
                 non_nodule_bmp_folder="Data/bmp/non_nodule",
                 csv_path="Data/CLNDAT_EN_df.csv"):
        """Initializes paths to data files."""
        self.__X_nodule_file_path = nodule_zip
        self.__X_non_nodule_file_path = non_nodule_zip
        self.__X_nodule_folder_path = nodule_bmp_folder
        self.__X_non_nodule_folder_path = non_nodule_bmp_folder
        self.__data_path = csv_path
        self.__img_dtype = np.uint16
        self.__img_width = 2048
        self.__img_height = 2048

    def create_X_y(self, y_type="", load_source="binary", color_mode="gray"):
        """
        Creates feature matrix (X) and target vector (y).

        Args:
            y_type (str): 'location' to get x,y coordinates as y, otherwise binary classification (0/1).
            load_source (str): 'binary' to load from .IMG files in zips, 'bmp' to load from BMP folders.
            color_mode (str): 'gray' for grayscale, 'rgb' for color (only affects BMP loading).

        Returns:
            tuple: (X, y) numpy arrays.
        """
        if load_source == "binary":
            X_nodule = self._load_binary_data(self.__X_nodule_file_path)
            if y_type != "location":
                X_non = self._load_binary_data(self.__X_non_nodule_file_path)
        elif load_source == "bmp":
            X_nodule = self._load_bmp_data(self.__X_nodule_folder_path, c=color_mode)
            if y_type != "location":
                X_non = self._load_bmp_data(self.__X_non_nodule_folder_path, c=color_mode)
        else:
            raise ValueError("load_source must be 'binary' or 'bmp'")

        if y_type == "location":
            X = X_nodule
            df = pd.read_csv(self.__data_path)
            # Ensure we only take labels for the nodule cases
            nodule_case_ids = [f"JPCLN{i:03d}" for i in range(1, 155)] # Assuming case IDs based on zip content
            nodule_df = df[df['caseID'].isin(nodule_case_ids)].reset_index()
            if len(nodule_df) != len(X_nodule):
                 print(f"Warning: Mismatch between number of nodule images ({len(X_nodule)}) and labels ({len(nodule_df)}). Ensure data alignment.")
                 # Adjust based on your actual data ordering/filtering needs
                 # This simple take might be incorrect if order differs:
                 nodule_df = nodule_df.iloc[:len(X_nodule)]

            x_c = nodule_df["x_coordinate"]
            y_c = nodule_df["y_coordinate"]
            y = np.column_stack((x_c, y_c))
        else:
            X = np.concatenate((X_nodule, X_non), axis=0)
            y_nodule = np.ones(X_nodule.shape[0], dtype=int)
            y_non = np.zeros(X_non.shape[0], dtype=int)
            y = np.concatenate((y_nodule, y_non), axis=0)

        return X, y

    def _load_binary_data(self, zip_file_path):
        """Loads .IMG data from a zip file."""
        image_data = []
        try:
            with zipfile.ZipFile(zip_file_path, "r") as archive:
                # Ensure consistent order by sorting filenames
                file_list = sorted([f for f in archive.namelist() if f.endswith('.IMG')])
                for filename in file_list:
                    with archive.open(filename) as img_file:
                        binary_data = img_file.read()
                        # Assuming big-endian based on original code hint ">u2"
                        img_array = np.frombuffer(binary_data, dtype=self.__img_dtype.newbyteorder('>'))
                        if img_array.shape[0] != self.__img_width * self.__img_height:
                             print(f"Warning: Unexpected image size in {filename}. Expected {self.__img_width * self.__img_height}, got {img_array.shape[0]}")
                             continue # Skip malformed files
                        image_data.append(img_array)
        except FileNotFoundError:
            print(f"Error: Zip file not found at {zip_file_path}")
            return np.array([])
        except Exception as e:
            print(f"An error occurred while reading {zip_file_path}: {e}")
            return np.array([])
        return np.array(image_data)


    def _load_bmp_data(self, folder_path, c="gray"):
        """Loads BMP data from a folder."""
        image_data = []
        if not os.path.isdir(folder_path):
             print(f"Error: Folder not found at {folder_path}")
             return np.array([])
             
        image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".bmp")]) # Sort for consistency
        
        for image_file in image_files:
            try:
                image_path = os.path.join(folder_path, image_file)
                image = cv.imread(image_path)
                if image is None:
                    print(f"Warning: Could not read image {image_file}. Skipping.")
                    continue

                expected_pixels = self.__img_width * self.__img_height
                
                if c == "gray":
                    if len(image.shape) > 2 and image.shape[2] == 3: # Check if it's color
                         image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
                    
                    if image.shape[0] != self.__img_height or image.shape[1] != self.__img_width:
                         print(f"Warning: Resizing {image_file} from {image.shape} to ({self.__img_height},{self.__img_width})")
                         image = cv.resize(image, (self.__img_width, self.__img_height)) # Resize if needed
                         
                    image = image.reshape(expected_pixels,)
                else: # Assuming RGB
                    if len(image.shape) == 2 or image.shape[2] == 1: # Check if it's grayscale
                         image = cv.cvtColor(image, cv.COLOR_GRAY2BGR) # Convert to color
                         
                    if image.shape[0] != self.__img_height or image.shape[1] != self.__img_width:
                         print(f"Warning: Resizing {image_file} from {image.shape} to ({self.__img_height},{self.__img_width})")
                         image = cv.resize(image, (self.__img_width, self.__img_height)) # Resize if needed
                    
                    expected_pixels *= 3 # Adjust for 3 channels
                    image = image.reshape(expected_pixels,)
                    
                image_data.append(image)
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
                continue # Skip file on error

        return np.array(image_data)
