import folium
import os
from folium import Icon, PolyLine
from folium.plugins import MeasureControl, Fullscreen  # Import plugins separately
from geopy.distance import geodesic

def create_map():
    # User's exact location coordinates
    user_lat = 13.127816848210971
    user_lon = 77.5865750885642

    # Charging stations near your location (sample data)
    charging_stations = [
        {'lat': 13.128466, 'lon': 77.587564, 'name': 'Tesla Supercharger'},
        {'lat': 13.126916, 'lon': 77.585264, 'name': 'EV Junction Hub'},
        {'lat': 13.129167, 'lon': 77.584575, 'name': 'ElectroPower Station'},
        {'lat': 13.125469, 'lon': 77.588712, 'name': 'GreenCharge Point'},
        {'lat': 13.127319, 'lon': 77.589465, 'name': 'VoltZone Charging'},
    ]

    # Create optimized map
    m = folium.Map(
        location=[user_lat, user_lon],
        zoom_start=17,
        tiles='cartodbpositron',
        control_scale=True,
        prefer_canvas=True
    )

    # Add precise user marker
    folium.Marker(
        location=[user_lat, user_lon],
        popup=folium.Popup(f"<b>Your Location</b><br>Lat: {user_lat}<br>Lon: {user_lon}", max_width=250),
        icon=Icon(color='red', icon='car', prefix='fa', icon_size=(24, 24)),
        z_index_offset=1000
    ).add_to(m)

    # Add charging stations with enhanced markers
    for idx, station in enumerate(charging_stations, 1):
        folium.Marker(
            location=[station['lat'], station['lon']],
            popup=folium.Popup(
                f"<b>{station['name']}</b><br>"
                f"Distance: {geodesic((user_lat, user_lon), (station['lat'], station['lon'])).m:.0f} meters",
                max_width=300
            ),
            icon=Icon(
                color='darkgreen',
                icon='bolt',
                prefix='fa',
                icon_color='white',
                icon_size=(18, 18)
            ),
            tooltip=f"Station {idx}"
        ).add_to(m)

        # Add connection line
        PolyLine(
            locations=[[user_lat, user_lon], [station['lat'], station['lon']]],
            color='#2c7be5',
            weight=1.5,
            opacity=0.7,
            dash_array='5,5'
        ).add_to(m)

    # Add map controls
    folium.LayerControl().add_to(m)
    MeasureControl(position='bottomleft').add_to(m)  # Now using directly imported MeasureControl
    Fullscreen(position='topright').add_to(m)  # Now using directly imported Fullscreen

    # Ensure static directory exists
    os.makedirs('static', exist_ok=True)

    # Save optimized map
    m.save('static/map.html')

create_map()