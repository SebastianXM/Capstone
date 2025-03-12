from flask import Flask, request
import json

compass_direction = 0

def create_app():
    app = Flask(__name__)

    @app.route('/data', methods=['POST'])
    def receive_data():
        global compass_direction
        try:
            data = request.get_json(force=True)
            payload = data.get('payload')

            for entry in payload:
                values = entry.get('values', {})
                if 'magneticBearing' in values:
                    compass_direction = values.get('magneticBearing')
            
            print("Flask Server Received direction:", compass_direction)
            return "200"
        except Exception as e:
            print("Error in receive_data:", e)
            return "400"

    return app

def run_server():
    app = create_app()
    app.run(host="192.168.1.48", port=8000, debug=False, use_reloader=False)
