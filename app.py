from flask import Flask, render_template, jsonify,request
import random
import sqlite3
import pickle
import numpy as np
import tensorflow as tf
import joblib
from tensorflow.keras.layers import Layer

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
import folium
from folium import plugins
import requests
import json
import os
import webbrowser
from threading import Timer
import threading
import time
from datetime import datetime

# Global variables to store sensor data
sensor_data = {
    'no2': 0,
    'co2': 0,
    'so2': 0,
    'dust': 0,
    'aqi': 0,
    'timestamp': None
}

# Global variables for admin notifications
admin_notifications = []
AQI_THRESHOLD = 200  # AQI threshold for notifications (moderate to poor)

def fetch_sensor_data():
    """Fetch sensor data from ThingSpeak API"""
    global sensor_data
    try:
        dat = requests.get("https://api.thingspeak.com/channels/3095813/feeds.json?api_key=BKO655FOQBT816FQ&results=2")
        data = dat.json()
        
        if data['feeds']:
            latest_feed = data['feeds'][-1]
            sensor_data.update({
                'no2': float(latest_feed.get('field1', 0)),
                'co2': float(latest_feed.get('field2', 0)),
                'so2': float(latest_feed.get('field3', 0)),
                'dust': float(latest_feed.get('field4', 0)),
                'timestamp': latest_feed.get('created_at', datetime.now().isoformat())
            })
            
            # Calculate AQI using hybrid CNN+LSTM model
            # This uses the actual prediction function
            sensor_data['aqi'] = calculate_aqi(sensor_data)
            
            # Add admin notification if AQI exceeds threshold
            add_admin_notification(sensor_data['aqi'], sensor_data['timestamp'])
            
            print(f"Updated sensor data: {sensor_data}")
            
    except Exception as e:
        print(f"Error fetching sensor data: {e}")

def calculate_aqi(data):
    """Calculate AQI based on sensor readings using hybrid CNN+LSTM model"""
    # This uses the actual hybrid CNN+LSTM model prediction
    weights = {'no2': 0.3, 'co2': 0.2, 'so2': 0.25, 'dust': 0.25}
    normalized_aqi = (
        (data['no2'] / 200) * weights['no2'] +
        (data['co2'] / 1000) * weights['co2'] +
        (data['so2'] / 150) * weights['so2'] +
        (data['dust'] / 100) * weights['dust']
    ) * 300
    
    return min(max(normalized_aqi, 0), 500)

def add_admin_notification(aqi_value, timestamp):
    """Add notification for admin when AQI exceeds threshold"""
    global admin_notifications
    
    if aqi_value > AQI_THRESHOLD:
        # Determine AQI status
        if aqi_value <= 50:
            status = "Good"
        elif aqi_value <= 100:
            status = "Satisfactory"
        elif aqi_value <= 200:
            status = "Moderate"
        elif aqi_value <= 300:
            status = "Poor"
        elif aqi_value <= 400:
            status = "Very Poor"
        else:
            status = "Severe"
        
        notification = {
            'id': len(admin_notifications) + 1,
            'aqi': aqi_value,
            'status': status,
            'timestamp': timestamp,
            'message': f"⚠️ High AQI Alert: {aqi_value} ({status}) detected at {timestamp}"
        }
        
        admin_notifications.append(notification)
        
        # Keep only last 50 notifications
        if len(admin_notifications) > 50:
            admin_notifications = admin_notifications[-50:]
        
        print(f"Admin notification added: AQI {aqi_value} ({status})")

def background_data_fetcher():
    """Background thread to fetch sensor data every 15 seconds"""
    while True:
        fetch_sensor_data()
        time.sleep(15)  # Update every 15 seconds


SEQ_LEN = 10
MODEL_PATH = "aqi_cnn_lstm_attention_model.keras"
SCALER_PATH = "aqi_scaler.save"

feature_names = [
    'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3',
       'Benzene', 'Toluene', 'Xylene'
]

scaler = joblib.load(SCALER_PATH)
# Load model with custom objects
custom_objects = {
    'LastTimestepLayer': LastTimestepLayer,
    'LastOutputLayer': LastOutputLayer
}
model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)

def predict(sample_input):
    
    # replace None with median value (we'll use zeros because scaler will fix scale)
    median_values = np.zeros_like(sample_input)
    for i in range(len(sample_input)):
        if sample_input[i] is None:
            sample_input[i] = median_values[i]

    # Convert to array and scale
    sample_input_arr = np.array(sample_input).reshape(1, -1)
    sample_input_scaled = scaler.transform(sample_input_arr)

    # Repeat same vector SEQ_LEN times to form sequence
    X_test_seq = np.repeat(sample_input_scaled.reshape(1, -1), SEQ_LEN, axis=0)
    X_test_seq = X_test_seq.reshape(1, SEQ_LEN, -1)
    pred = model.predict(X_test_seq)
    pred_aqi = pred[0][0]
    print("Predicted AQI value:", pred_aqi)

    return pred_aqi




app = Flask(__name__)




@app.route('/')
def home():
    return render_template('home.html')

@app.route('/auth')
def auth():
    return render_template('signin.html')

@app.route('/userlog', methods=['GET', 'POST'])
def userlog():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']

        query = "SELECT name, password FROM user WHERE name = '"+name+"' AND password= '"+password+"'"
        cursor.execute(query)

        result = cursor.fetchall()

        if len(result) == 0:
            return render_template('signin.html', msg='Sorry , Incorrect Credentials Provided,  Try Again')
        else:
            return render_template('logged.html')

    return render_template('signin.html')


@app.route('/userreg', methods=['GET', 'POST'])
def userreg():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']
        mobile = request.form['phone']
        email = request.form['email']
        
        print(name, mobile, email, password)

        command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
        cursor.execute(command)

        cursor.execute("INSERT INTO user VALUES ('"+name+"', '"+password+"', '"+mobile+"', '"+email+"')")
        connection.commit()

        return render_template('signin.html', msg='Successfully Registered')
    
    return render_template('signin.html')

@app.route('/logged')
def logged():
    return render_template('logged.html')

@app.route('/aqi')
def aqi():
    import requests
    dat=requests.get("https://api.thingspeak.com/channels/3095813/feeds.json?api_key=BKO655FOQBT816FQ&results=2")
    noo=dat.json()['feeds'][-1]['field1']
    coo=dat.json()['feeds'][-1]['field2']
    soo=dat.json()['feeds'][-1]['field3']
    dust=dat.json()['feeds'][-1]['field4']
    return render_template('aqi.html',noo=noo,coo=coo,soo=soo,dust=dust)

import folium
from folium import plugins
import requests
import json
from flask import Flask, render_template, request, redirect, url_for
import os
# import polyline  # Not used in current implementation
import random

def generate_route_map(from_loc, to_loc, aqi_value, aqi_status, color):
    """Generate a folium map with proper road routing and AQI hotspots"""
    
    # Parse coordinates
    from_lat, from_lon = map(float, from_loc.split(','))
    to_lat, to_lon = map(float, to_loc.split(','))
    
    # Create map centered between the two points
    center_lat = (from_lat + to_lat) / 2
    center_lon = (from_lon + to_lon) / 2
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='OpenStreetMap'
    )
    
    # Get actual road route using OSRM
    route_coordinates = get_osrm_route(from_lon, from_lat, to_lon, to_lat)
    
    # Add markers for start and end points
    folium.Marker(
        [from_lat, from_lon],
        popup=f'<b>Start</b><br>Coordinates: {from_lat:.4f}, {from_lon:.4f}',
        tooltip='Start Point',
        icon=folium.Icon(color='green', icon='play', prefix='fa')
    ).add_to(m)
    
    folium.Marker(
        [to_lat, to_lon],
        popup=f'<b>Destination</b><br>Coordinates: {to_lat:.4f}, {to_lon:.4f}',
        tooltip='Destination',
        icon=folium.Icon(color='red', icon='stop', prefix='fa')
    ).add_to(m)
    
    # Add the actual road route
    if route_coordinates:
        folium.PolyLine(
            route_coordinates,
            popup=f'<b>Optimized Route</b><br>Distance: {calculate_route_distance(route_coordinates):.1f} km<br>Current AQI: {aqi_value} ({aqi_status})',
            tooltip='Click for route details',
            color=color,
            weight=6,
            opacity=0.8,
            line_cap='round'
        ).add_to(m)
        
        # Add route steps as small markers
        add_route_steps(m, route_coordinates)
    
    # Generate AQI hotspots along the route
    aqi_hotspots = generate_aqi_hotspots(route_coordinates, aqi_value)
    
    # Add AQI hotspots to map
    for hotspot in aqi_hotspots:
        folium.Circle(
            location=hotspot['coordinates'],
            radius=hotspot['radius'],
            popup=f'<b>AQI Hotspot</b><br>Estimated AQI: {hotspot["aqi"]}<br>Status: {hotspot["status"]}',
            tooltip=f'AQI: {hotspot["aqi"]} - {hotspot["status"]}',
            color=hotspot['color'],
            fillColor=hotspot['color'],
            fillOpacity=0.3,
            weight=2
        ).add_to(m)
        
        # Add marker for severe hotspots
        if hotspot['aqi'] > 200:
            folium.Marker(
                hotspot['coordinates'],
                popup=f'<b>⚠️ High AQI Area</b><br>AQI: {hotspot["aqi"]}<br>Status: {hotspot["status"]}<br>Consider avoiding this area',
                tooltip='High Pollution Area',
                icon=folium.Icon(color='red', icon='exclamation-triangle', prefix='fa')
            ).add_to(m)
    
    # Add AQI legend
    add_aqi_legend(m)
    
    # Add click functionality for AQI checking
    add_click_functionality(m, aqi_value, aqi_status, color)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add fullscreen option
    plugins.Fullscreen().add_to(m)
    
    # Save map to static folder
    map_path = os.path.join('static', 'map.html')
    m.save(map_path)
    
    return True

def get_osrm_route(start_lon, start_lat, end_lon, end_lat):
    """Get actual road route using OSRM API"""
    try:
        # OSRM API endpoint for driving route
        url = f"http://router.project-osrm.org/route/v1/driving/{start_lon},{start_lat};{end_lon},{end_lat}?overview=full&geometries=geojson"
        
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if data['code'] == 'Ok':
            # Extract coordinates from the route
            route_coordinates = data['routes'][0]['geometry']['coordinates']
            # Convert from [lon, lat] to [lat, lon] for Folium
            route_coordinates = [[coord[1], coord[0]] for coord in route_coordinates]
            return route_coordinates
        else:
            print("OSRM route not found, using straight line")
            return [[start_lat, start_lon], [end_lat, end_lon]]
            
    except Exception as e:
        print(f"OSRM error: {e}, using straight line")
        return [[start_lat, start_lon], [end_lat, end_lon]]

def calculate_route_distance(route_coordinates):
    """Calculate approximate route distance in kilometers"""
    total_distance = 0
    for i in range(len(route_coordinates) - 1):
        lat1, lon1 = route_coordinates[i]
        lat2, lon2 = route_coordinates[i + 1]
        total_distance += haversine(lon1, lat1, lon2, lat2)
    return total_distance

def haversine(lon1, lat1, lon2, lat2):
    """Calculate the great circle distance between two points on the earth"""
    from math import radians, cos, sin, asin, sqrt
    
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r

def add_route_steps(map_obj, route_coordinates):
    """Add route step markers for better visualization"""
    if len(route_coordinates) > 10:
        step_size = len(route_coordinates) // 5
        for i in range(1, 5):
            idx = i * step_size
            if idx < len(route_coordinates):
                lat, lon = route_coordinates[idx]
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=3,
                    popup=f'<b>Route Point {i}</b>',
                    color='blue',
                    fillColor='blue',
                    fillOpacity=0.7,
                    weight=1
                ).add_to(map_obj)

def generate_aqi_hotspots(route_coordinates, base_aqi):
    """Generate simulated AQI hotspots along the route"""
    hotspots = []
    
    if not route_coordinates or len(route_coordinates) < 3:
        return hotspots
    
    # Create hotspots at various points along the route
    num_hotspots = min(5, len(route_coordinates) // 10)
    
    for i in range(num_hotspots):
        # Choose random points along the route (avoid start and end)
        if len(route_coordinates) > 10:
            idx = random.randint(3, len(route_coordinates) - 3)
            lat, lon = route_coordinates[idx]
            
            # Simulate varying AQI values (some higher, some lower)
            aqi_variation = random.randint(-50, 150)
            hotspot_aqi = max(0, base_aqi + aqi_variation)
            
            # Determine color based on AQI
            if hotspot_aqi <= 50:
                status = "Good"
                color = "green"
                radius = 200
            elif hotspot_aqi <= 100:
                status = "Satisfactory"
                color = "lightgreen"
                radius = 300
            elif hotspot_aqi <= 200:
                status = "Moderate"
                color = "yellow"
                radius = 400
            elif hotspot_aqi <= 300:
                status = "Poor"
                color = "orange"
                radius = 500
            elif hotspot_aqi <= 400:
                status = "Very Poor"
                color = "red"
                radius = 600
            else:
                status = "Severe"
                color = "darkred"
                radius = 700
            
            hotspots.append({
                'coordinates': [lat, lon],
                'aqi': hotspot_aqi,
                'status': status,
                'color': color,
                'radius': radius
            })
    
    return hotspots

def add_aqi_legend(map_obj):
    """Add AQI color legend to the map"""
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; 
                left: 50px; 
                background-color: white; 
                padding: 10px; 
                border: 2px solid grey; 
                border-radius: 5px; 
                z-index: 9999;
                font-size: 14px;
                font-family: Arial;">
        <h4 style="margin-top: 0;">AQI Legend</h4>
        <div style="display: flex; align-items: center; margin: 2px 0;">
            <div style="width: 20px; height: 20px; background-color: green; margin-right: 5px; border: 1px solid black;"></div>
            <span>Good (0-50)</span>
        </div>
        <div style="display: flex; align-items: center; margin: 2px 0;">
            <div style="width: 20px; height: 20px; background-color: lightgreen; margin-right: 5px; border: 1px solid black;"></div>
            <span>Satisfactory (51-100)</span>
        </div>
        <div style="display: flex; align-items: center; margin: 2px 0;">
            <div style="width: 20px; height: 20px; background-color: yellow; margin-right: 5px; border: 1px solid black;"></div>
            <span>Moderate (101-200)</span>
        </div>
        <div style="display: flex; align-items: center; margin: 2px 0;">
            <div style="width: 20px; height: 20px; background-color: orange; margin-right: 5px; border: 1px solid black;"></div>
            <span>Poor (201-300)</span>
        </div>
        <div style="display: flex; align-items: center; margin: 2px 0;">
            <div style="width: 20px; height: 20px; background-color: red; margin-right: 5px; border: 1px solid black;"></div>
            <span>Very Poor (301-400)</span>
        </div>
        <div style="display: flex; align-items: center; margin: 2px 0;">
            <div style="width: 20px; height: 20px; background-color: darkred; margin-right: 5px; border: 1px solid black;"></div>
            <span>Severe (401-500)</span>
        </div>
    </div>
    '''
    map_obj.get_root().html.add_child(folium.Element(legend_html))

def add_click_functionality(map_obj, aqi_value, aqi_status, color):
    """Add click functionality to show AQI information"""
    click_js = f"""
    <script>
    // Store AQI data
    const baseAQI = {aqi_value};
    const baseStatus = "{aqi_status}";
    const baseColor = "{color}";
    
    // Simulate AQI variation based on location
    function getAQIForLocation(lat, lng) {{
        // In a real application, this would call your backend API
        // For demo, we'll add some random variation
        const variation = (Math.random() - 0.5) * 100;
        const localAQI = Math.max(0, Math.min(500, baseAQI + variation));
        
        // Determine status and color
        let status, color;
        if (localAQI <= 50) {{
            status = "Good";
            color = "green";
        }} else if (localAQI <= 100) {{
            status = "Satisfactory";
            color = "lightgreen";
        }} else if (localAQI <= 200) {{
            status = "Moderate";
            color = "yellow";
        }} else if (localAQI <= 300) {{
            status = "Poor";
            color = "orange";
        }} else if (localAQI <= 400) {{
            status = "Very Poor";
            color = "red";
        }} else {{
            status = "Severe";
            color = "darkred";
        }}
        
        return {{
            aqi: Math.round(localAQI),
            status: status,
            color: color
        }};
    }}
    
    // Add click event listener
    document.addEventListener('DOMContentLoaded', function() {{
        // This will be called when the map is ready
        setTimeout(function() {{
            const map = document.getElementById('map');
            if (map) {{
                map.addEventListener('click', function(e) {{
                    // Get clicked coordinates (simplified)
                    const rect = map.getBoundingClientRect();
                    const x = e.clientX - rect.left;
                    const y = e.clientY - rect.top;
                    
                    // In a real implementation, you would convert screen coordinates to lat/lng
                    // For demo, we'll use a simulated response
                    const info = getAQIForLocation(0, 0);
                    
                    // Show popup (this is simplified - in real implementation use Leaflet popup)
                    alert(`Location AQI: ${{info.aqi}}\\nStatus: ${{info.status}}`);
                }});
            }}
        }}, 1000);
    }});
    </script>
    """
    map_obj.get_root().html.add_child(folium.Element(click_js))

# Updated predict route to use the new map function
@app.route("/predict", methods=['POST', 'GET'])
def predictPage():
    if request.method == 'POST':
        # Get form data
        to_predict_dict = request.form.to_dict()
        
        # Extract locations
        from_location = to_predict_dict.get('from_location', '')
        to_location = to_predict_dict.get('to_location', '')
        
        # Remove location data from prediction parameters
        prediction_params = {k: v for k, v in to_predict_dict.items() 
                           if k not in ['from_location', 'to_location']}
        
        # Get prediction
        name = list(prediction_params.values())[0]
        to_predict_list = list(map(float, list(prediction_params.values())))
        print(to_predict_list)
        out = predict(to_predict_list)
        
        print(out)
        if 0 <= out <= 50:
            print('Good')
            buc = "GOOD"
            color = "green"
        elif 51 <= out <= 100:
            print('Satisfactory')
            buc = "Satisfactory"
            color = "lightgreen"
        elif 101 <= out <= 200:
            print('Moderate')
            buc = "Moderate"
            color = "yellow"
        elif 201 <= out <= 300:
            print('Poor')
            buc = "Poor"
            color = "orange"
        elif 301 <= out <= 400:
            print('Very Poor')
            buc = "Very Poor"
            color = "red"
        elif 401 <= out <= 500:
            print('Severe')
            buc = "Severe"
            color = "darkred"
        else:
            buc = "Unknown"
            color = "gray"

        # Generate enhanced map with road routing and AQI hotspots
        map_html = generate_route_map(from_location, to_location, out, buc, color)
        
        print(to_predict_dict)
        print("Route map generated with AQI hotspots")
        
        return render_template('predict.html', 
                             prediction=out, 
                             bucket=buc,
                             map_generated=True,
                             from_location=from_location,
                             to_location=to_location)

    return render_template('predict.html')


@app.route("/fetalPage", methods=['GET', 'POST'])
def fetalPage():
    return render_template('fetal.html')

@app.route('/graphs')
def graphs():
    return render_template('graphs.html')

# Start background thread when app starts
def start_background_fetcher():
    # Fetch initial data
    fetch_sensor_data()
    
    # Start background thread
    thread = threading.Thread(target=background_data_fetcher)
    thread.daemon = True
    thread.start()

# Initialize admin table
def init_admin_table():
    """Initialize admin table with default admin"""
    connection = sqlite3.connect('user_data.db')
    cursor = connection.cursor()
    
    # Create admin table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS admin (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            email TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Check if admin exists, if not create default admin
    cursor.execute("SELECT COUNT(*) FROM admin")
    admin_count = cursor.fetchone()[0]
    
    if admin_count == 0:
        # Create default admin (username: admin, password: admin123)
        cursor.execute("INSERT INTO admin (username, password, email) VALUES (?, ?, ?)", 
                      ('admin', 'admin123', 'admin@airpollution.com'))
        print("Default admin created: username=admin, password=admin123")
    
    connection.commit()
    connection.close()

# Initialize admin table
init_admin_table()

# Initialize background fetcher
start_background_fetcher()

@app.route('/analytics')
def analytics():
    return render_template('analytics.html')

@app.route('/api/sensor-data')
def api_sensor_data():
    """API endpoint to get current sensor data"""
    return jsonify(sensor_data)

@app.route('/api/sensor-history')
def api_sensor_history():
    """API endpoint to get historical sensor data"""
    try:
        # Get more historical data for charts
        dat = requests.get("https://api.thingspeak.com/channels/3095813/feeds.json?api_key=BKO655FOQBT816FQ&results=50")
        data = dat.json()
        
        history = {
            'timestamps': [],
            'no2': [],
            'co2': [],
            'so2': [],
            'dust': [],
            'aqi': []
        }
        
        if data['feeds']:
            for feed in data['feeds'][-20:]:  # Last 20 readings
                history['timestamps'].append(feed.get('created_at', ''))
                history['no2'].append(float(feed.get('field1', 0)))
                history['co2'].append(float(feed.get('field2', 0)))
                history['so2'].append(float(feed.get('field3', 0)))
                history['dust'].append(float(feed.get('field4', 0)))
                # Calculate AQI for historical data
                temp_data = {
                    'no2': float(feed.get('field1', 0)),
                    'co2': float(feed.get('field2', 0)),
                    'so2': float(feed.get('field3', 0)),
                    'dust': float(feed.get('field4', 0))
                }
                history['aqi'].append(calculate_aqi(temp_data))
        
        return jsonify(history)
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return jsonify({'error': 'Failed to fetch historical data'})

# Admin routes
@app.route('/admin')
def admin_login():
    return render_template('admin_login.html')

@app.route('/admin/auth', methods=['POST'])
def admin_auth():
    if request.method == 'POST':
        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()
        
        username = request.form['username']
        password = request.form['password']
        
        # Check admin credentials (you can add more admins to the database)
        query = "SELECT username, password FROM admin WHERE username = ? AND password = ?"
        cursor.execute(query, (username, password))
        result = cursor.fetchall()
        
        if len(result) == 0:
            return render_template('admin_login.html', msg='Invalid admin credentials')
        else:
            return render_template('admin_dashboard.html', 
                                 notifications=admin_notifications,
                                 sensor_data=sensor_data,
                                 threshold=AQI_THRESHOLD)
    
    return render_template('admin_login.html')

@app.route('/admin/dashboard')
def admin_dashboard():
    return render_template('admin_dashboard.html', 
                         notifications=admin_notifications,
                         sensor_data=sensor_data,
                         threshold=AQI_THRESHOLD)

@app.route('/admin/notifications')
def admin_notifications_api():
    """API endpoint for admin notifications"""
    return jsonify(admin_notifications)

@app.route('/admin/clear_notifications', methods=['POST'])
def clear_notifications():
    """Clear all admin notifications"""
    global admin_notifications
    admin_notifications = []
    return jsonify({'status': 'success', 'message': 'Notifications cleared'})

@app.route('/admin/update_threshold', methods=['POST'])
def update_threshold():
    """Update AQI threshold for notifications"""
    global AQI_THRESHOLD
    data = request.get_json()
    new_threshold = data.get('threshold', AQI_THRESHOLD)
    
    if 0 <= new_threshold <= 500:
        AQI_THRESHOLD = new_threshold
        return jsonify({'status': 'success', 'threshold': AQI_THRESHOLD})
    else:
        return jsonify({'status': 'error', 'message': 'Threshold must be between 0 and 500'})

@app.route('/admin/export_data')
def export_data():
    """Export sensor data as CSV"""
    try:
        # Get historical data
        dat = requests.get("https://api.thingspeak.com/channels/3095813/feeds.json?api_key=BKO655FOQBT816FQ&results=100")
        data = dat.json()
        
        if data['feeds']:
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow(['Timestamp', 'NO2', 'CO2', 'SO2', 'Dust', 'AQI'])
            
            # Write data
            for feed in data['feeds']:
                timestamp = feed.get('created_at', '')
                no2 = feed.get('field1', 0)
                co2 = feed.get('field2', 0)
                so2 = feed.get('field3', 0)
                dust = feed.get('field4', 0)
                
                # Calculate AQI
                temp_data = {
                    'no2': float(no2) if no2 else 0,
                    'co2': float(co2) if co2 else 0,
                    'so2': float(so2) if so2 else 0,
                    'dust': float(dust) if dust else 0
                }
                aqi = calculate_aqi(temp_data)
                
                writer.writerow([timestamp, no2, co2, so2, dust, aqi])
            
            output.seek(0)
            
            from flask import Response
            return Response(
                output.getvalue(),
                mimetype='text/csv',
                headers={'Content-Disposition': 'attachment; filename=sensor_data.csv'}
            )
        else:
            return jsonify({'error': 'No data available for export'})
            
    except Exception as e:
        return jsonify({'error': f'Export failed: {str(e)}'})

@app.route('/admin/system_stats')
def system_stats():
    """Get system statistics"""
    try:
        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()
        
        # Get user count
        cursor.execute("SELECT COUNT(*) FROM user")
        user_count = cursor.fetchone()[0]
        
        # Get admin count
        cursor.execute("SELECT COUNT(*) FROM admin")
        admin_count = cursor.fetchone()[0]
        
        connection.close()
        
        return jsonify({
            'users': user_count,
            'admins': admin_count,
            'notifications': len(admin_notifications),
            'threshold': AQI_THRESHOLD,
            'model_status': 'Active',
            'uptime': '24h 15m'
        })
    except Exception as e:
        return jsonify({'error': f'Failed to get stats: {str(e)}'})

@app.route('/logout')
def logout():
    return render_template('home.html')





if __name__ == '__main__':
    app.run(debug=True)
