# 3. streaming_server.py
import asyncio
import json
from aiohttp import web
from model_loader import ModelLoader
from anomaly_detector import AnomalyDetector

class StreamingServer:
    def __init__(self):
        self.model_loader = ModelLoader()
        self.anomaly_detector = AnomalyDetector()
        self.current_model = None

    async def handle_data(self, request):
        data = await request.json()
        # Process incoming data
        # Detect anomalies using the current model
        is_anomaly = self.anomaly_detector.detect(self.current_model, data)
        if is_anomaly:
            await self.send_anomaly_alert()
        return web.Response(text="Data received")

    async def change_model(self, request):
        model_name = await request.text()
        self.current_model = self.model_loader.load_model(model_name)
        return web.Response(text=f"Model changed to {model_name}")

    async def send_anomaly_alert(self):
        # Send alert to visualization server
        # Implementation depends on the communication protocol with the visualization server
        pass

    def run(self):
        app = web.Application()
        app.router.add_post('/data', self.handle_data)
        app.router.add_post('/change_model', self.change_model)
        web.run_app(app)

if __name__ == "__main__":
    server = StreamingServer()
    server.run()