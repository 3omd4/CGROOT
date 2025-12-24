from PyQt6.QtGui import QImage, qRgb
import numpy as np

class VisualizationManager:
    @staticmethod
    def create_preview_image(img_data_vec, width, height, depth):
        """
        Creates a QImage from raw pixel data vector (list of ints/uint8).
        Handles Grayscale (depth=1) and RGB (depth=3).
        """
        if not img_data_vec:
            return None

        # Check expected size
        expected_size = width * height * depth
        if len(img_data_vec) != expected_size:
            # print(f"Warning: Image data length mismatch. Expected {expected_size}, got {len(img_data_vec)}")
            return None

        try:
            if depth == 1:
                q_img = QImage(width, height, QImage.Format.Format_Grayscale8)
                # Optimization: In Python, looping setPixel is slow. 
                # Ideally, we construct bytes and use QImage(bytes, w, h, fmt)
                # But PyBind11 vector<uint8> comes as list[int].
                
                # Fast path using bytes
                data_bytes = bytes(img_data_vec)
                q_img = QImage(data_bytes, width, height, width, QImage.Format.Format_Grayscale8).copy()
                
                # Slow path (fallback logic if bytes fails or for reference)
                # for y in range(height):
                #     for x in range(width):
                #         val = img_data_vec[y * width + x]
                #         q_img.setPixel(x, y, qRgb(val, val, val))
                return q_img

            elif depth == 3:
                # CGROOT Format is Planar (RRR...GGG...BBB...)
                # QImage RGB888 expects Interleaved (RGB RGB RGB...)
                # We need to interleave.
                
                area = width * height
                r_start = 0
                g_start = area
                b_start = 2 * area
                
                # This loop is still potentially slow in pure Python, but required for Planar -> Interleaved
                # Optimization: Use numpy if available, but staying dependency-light for now
                # except we already import numpy above? No, I added import numpy but project didn't use it before.
                # Let's stick to list comprehension which is faster than setPixel loop
                
                # Create interleaved bytes
                # This is essentially: zip(R, G, B) -> flatten
                R = img_data_vec[r_start:r_start+area]
                G = img_data_vec[g_start:g_start+area]
                B = img_data_vec[b_start:b_start+area]
                
                # Interleave
                interleaved = []
                for r, g, b in zip(R, G, B):
                    interleaved.extend([r, g, b])
                    
                data_bytes = bytes(interleaved)
                q_img = QImage(data_bytes, width, height, width * 3, QImage.Format.Format_RGB888).copy()
                return q_img
            
            else:
                return None

        except Exception as e:
            print(f"Error creating preview image: {e}")
            return None

    @staticmethod
    def process_feature_maps(maps, layer_type):
        """
        Process feature maps for display if needed. 
        Currently just pass-through, but good place for normalization logic.
        """
        return maps
