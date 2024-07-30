import cv2
import pandas as pd
import os

def augment_wildfire_images(image_dir, coords_wgs84):
    latitudes = [coords_wgs84[1], coords_wgs84[3]]  
    longitudes = [coords_wgs84[0], coords_wgs84[2]]

    for image_path in os.listdir(os.path.join(image_dir, "masked")):
        full_path = os.path.join(image_dir, "masked", image_path)
        image = cv2.imread(full_path)
        # Extract image dimensions
        height, width, _ = image.shape
        data = []
        for i in range(height):
            for j in range(width):
                pixel = image[i, j]
                if (pixel == [0, 0, 255]).all():  # bgr
                    latitude = latitudes[0] + (i / height) * (latitudes[1] - latitudes[0])
                    longitude = longitudes[0] + (j / width) * (longitudes[1] - longitudes[0])
                    data.append([latitude, longitude, "forestfire"])
        # Save data to JSON
        if data:
            df = pd.DataFrame(data, columns=['latitude', 'longitude', 'status'])
            jsonpath = os.path.join(image_dir, "augmented_coordinates", image_path.replace(".png", ".json"))
            os.makedirs(os.path.dirname(jsonpath), exist_ok=True)
            df.to_json(jsonpath, orient="index") 
            #data_name = str(os.path.splitext(image_path)[0]).strip("(").replace(")","").replace("-","_").replace(",","_").replace("\'","").replace(" ","_")
           
