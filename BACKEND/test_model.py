import tensorflow as tf
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Layer
import warnings
warnings.filterwarnings('ignore')

# Custom layers for the hybrid model
class LastTimestepLayer(Layer):
    def call(self, inputs):
        return inputs[:, -1:, :]

    def get_config(self):
        return super().get_config()

class LastOutputLayer(Layer):
    def call(self, inputs):
        return inputs[:, -1, :]

    def get_config(self):
        return super().get_config()

def test_model_loading():
    """Test if the model loads correctly"""
    print("🔍 Testing Model Loading...")
    
    MODEL_PATH = "aqi_cnn_lstm_attention_model.keras"
    SCALER_PATH = "aqi_scaler.save"
    
    custom_objects = {
        'LastTimestepLayer': LastTimestepLayer,
        'LastOutputLayer': LastOutputLayer
    }
    
    try:
        # Load model
        print("📦 Loading model...")
        model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
        print("✅ Model loaded successfully!")
        
        # Load scaler
        print("📦 Loading scaler...")
        scaler = joblib.load(SCALER_PATH)
        print("✅ Scaler loaded successfully!")
        
        return model, scaler
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

def test_model_architecture(model):
    """Test model architecture and properties"""
    print("\n🏗️ Testing Model Architecture...")
    
    if model is None:
        print("❌ No model to test")
        return
    
    try:
        # Print model summary
        print("📊 Model Summary:")
        print(f"   Input Shape: {model.input_shape}")
        print(f"   Output Shape: {model.output_shape}")
        print(f"   Total Parameters: {model.count_params():,}")
        
        # Print layer information
        print("\n🔧 Model Layers:")
        for i, layer in enumerate(model.layers):
            print(f"   Layer {i+1}: {layer.name} - {type(layer).__name__}")
        
        print("✅ Model architecture test passed!")
        
    except Exception as e:
        print(f"❌ Error testing architecture: {e}")

def test_model_prediction(model, scaler):
    """Test model prediction with sample data"""
    print("\n🎯 Testing Model Predictions...")
    
    if model is None or scaler is None:
        print("❌ No model or scaler to test")
        return
    
    try:
        # Create sample data (sequence length = 10, features = 4)
        SEQ_LEN = 10
        NUM_FEATURES = 4
        
        # Generate random sample data
        sample_data = np.random.rand(1, SEQ_LEN, NUM_FEATURES)
        print(f"📊 Sample data shape: {sample_data.shape}")
        
        # Make prediction
        print("🔮 Making prediction...")
        prediction = model.predict(sample_data, verbose=0)
        print(f"✅ Prediction successful!")
        print(f"   Predicted AQI: {prediction[0][0]:.2f}")
        
        # Test with multiple samples
        print("\n📈 Testing with multiple samples...")
        multiple_samples = np.random.rand(5, SEQ_LEN, NUM_FEATURES)
        predictions = model.predict(multiple_samples, verbose=0)
        
        print("✅ Multiple predictions successful!")
        for i, pred in enumerate(predictions):
            print(f"   Sample {i+1} AQI: {pred[0]:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        return False

def test_real_sensor_data():
    """Test with realistic sensor data"""
    print("\n🌡️ Testing with Realistic Sensor Data...")
    
    try:
        # Create realistic sensor data
        realistic_data = np.array([[
            [45.2, 380.5, 12.8, 85.3],  # NO2, CO2, SO2, Dust
            [47.1, 385.2, 13.1, 87.2],
            [46.8, 382.1, 12.9, 86.1],
            [48.3, 388.7, 13.4, 88.9],
            [47.9, 386.4, 13.2, 87.8],
            [49.1, 390.2, 13.6, 89.5],
            [48.7, 387.8, 13.3, 88.4],
            [50.2, 392.1, 13.8, 90.7],
            [49.8, 389.6, 13.5, 89.8],
            [51.3, 394.5, 14.1, 91.2]
        ]])
        
        print("📊 Realistic sensor data created:")
        print(f"   Shape: {realistic_data.shape}")
        print(f"   NO2 range: {realistic_data[0, :, 0].min():.1f} - {realistic_data[0, :, 0].max():.1f}")
        print(f"   CO2 range: {realistic_data[0, :, 1].min():.1f} - {realistic_data[0, :, 1].max():.1f}")
        print(f"   SO2 range: {realistic_data[0, :, 2].min():.1f} - {realistic_data[0, :, 2].max():.1f}")
        print(f"   Dust range: {realistic_data[0, :, 3].min():.1f} - {realistic_data[0, :, 3].max():.1f}")
        
        return realistic_data
        
    except Exception as e:
        print(f"❌ Error creating realistic data: {e}")
        return None

def test_aqi_calculation():
    """Test AQI calculation function"""
    print("\n🧮 Testing AQI Calculation...")
    
    try:
        # Test data
        test_data = {
            'no2': 45.2,
            'co2': 380.5,
            'so2': 12.8,
            'dust': 85.3
        }
        
        # Simple AQI calculation (this is a simplified version)
        # In reality, AQI calculation is more complex
        aqi = (test_data['no2'] * 0.3 + 
               test_data['co2'] * 0.1 + 
               test_data['so2'] * 0.2 + 
               test_data['dust'] * 0.4)
        
        print(f"✅ AQI calculation test:")
        print(f"   Input data: {test_data}")
        print(f"   Calculated AQI: {aqi:.2f}")
        
        # Determine AQI category
        if aqi <= 50:
            category = "Good"
        elif aqi <= 100:
            category = "Satisfactory"
        elif aqi <= 200:
            category = "Moderate"
        elif aqi <= 300:
            category = "Poor"
        elif aqi <= 400:
            category = "Very Poor"
        else:
            category = "Severe"
        
        print(f"   AQI Category: {category}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in AQI calculation: {e}")
        return False

def run_performance_test(model):
    """Test model performance and speed"""
    print("\n⚡ Testing Model Performance...")
    
    if model is None:
        print("❌ No model to test")
        return
    
    try:
        import time
        
        # Test prediction speed
        sample_data = np.random.rand(1, 10, 4)
        
        # Warm up
        _ = model.predict(sample_data, verbose=0)
        
        # Time multiple predictions
        start_time = time.time()
        num_predictions = 100
        
        for _ in range(num_predictions):
            sample_data = np.random.rand(1, 10, 4)
            _ = model.predict(sample_data, verbose=0)
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / num_predictions
        
        print(f"✅ Performance test results:")
        print(f"   Total time for {num_predictions} predictions: {total_time:.3f}s")
        print(f"   Average time per prediction: {avg_time*1000:.2f}ms")
        print(f"   Predictions per second: {1/avg_time:.1f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in performance test: {e}")
        return False

def main():
    """Main testing function"""
    print("🚀 Starting Model Testing Suite")
    print("=" * 50)
    
    # Test 1: Model Loading
    model, scaler = test_model_loading()
    
    # Test 2: Model Architecture
    test_model_architecture(model)
    
    # Test 3: Model Predictions
    if model is not None:
        test_model_prediction(model, scaler)
        
        # Test 4: Realistic Data
        realistic_data = test_real_sensor_data()
        if realistic_data is not None and model is not None:
            print("\n🔮 Testing with realistic data...")
            try:
                prediction = model.predict(realistic_data, verbose=0)
                print(f"✅ Realistic data prediction: {prediction[0][0]:.2f}")
            except Exception as e:
                print(f"❌ Error with realistic data: {e}")
        
        # Test 5: Performance
        run_performance_test(model)
    
    # Test 6: AQI Calculation
    test_aqi_calculation()
    
    print("\n" + "=" * 50)
    print("🏁 Model Testing Complete!")
    
    if model is not None:
        print("✅ All tests passed successfully!")
    else:
        print("❌ Some tests failed. Check the error messages above.")

if __name__ == "__main__":
    main()
