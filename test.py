import json
import re
from app import app, NUMERIC_FEATURES, BINARY_FEATURES, trained

def test_prediction():
    with app.test_client() as client:
        # Test 1: Empty form (should use medians)
        response = client.post("/predict", data={})
        html = response.data.decode("utf-8")
        
        # Extract prediction from HTML
        match = re.search(r'text-4xl[^>]*>\$([\d,]+)</div>', html)
        if match:
            print(f"Empty Form Prediction: ${match.group(1)}")
        else:
            print("Failed to find prediction in Test 1")
        
        # Test 2: Specific values
        data = {
            "city": "Saskatoon",
            "bedrooms": "3",
            "bathrooms": "2",
            "size_interior_sqft": "1200",
            "lot_size_sqft": "5000",
            "year_built": "2000",
            "parking_spaces": "2",
            "is_house": "1",
            "is_condo": "0",
            "has_garage": "1",
            "has_basement": "1",
            "basement_finished": "1",
            "has_cooling": "1"
        }
        response2 = client.post("/predict", data=data)
        html2 = response2.data.decode("utf-8")
        match2 = re.search(r'text-4xl[^>]*>\$([\d,]+)</div>', html2)
        if match2:
            print(f"Test 2 (Typical House) Prediction: ${match2.group(1)}")

        # Test 3: Extremely large lot size (to verify percentile maxes out properly)
        data["lot_size_sqft"] = "1000000"
        response3 = client.post("/predict", data=data)
        html3 = response3.data.decode("utf-8")
        match3 = re.search(r'text-4xl[^>]*>\$([\d,]+)</div>', html3)
        if match3:
            print(f"Test 3 (Huge Lot) Prediction: ${match3.group(1)}")
            
        # Test 4: Extremely small lot size
        data["lot_size_sqft"] = "10"
        response4 = client.post("/predict", data=data)
        html4 = response4.data.decode("utf-8")
        match4 = re.search(r'text-4xl[^>]*>\$([\d,]+)</div>', html4)
        if match4:
            print(f"Test 4 (Tiny Lot) Prediction: ${match4.group(1)}")

if __name__ == "__main__":
    test_prediction()
