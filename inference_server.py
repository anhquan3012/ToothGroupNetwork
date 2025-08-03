import asyncio
import websockets
import json
from inference_tgnet import inference_tgnet

async def handle_connection(websocket):
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                response = {"status": "error", "message": "Invalid JSON format."}
                await websocket.send(json.dumps(response))
                continue
            
            lower_scan = data.get("lower_scan")
            upper_scan = data.get("upper_scan")
            output_dir = data.get("output_dir")

            if (lower_scan != 'null' or upper_scan != 'null') and output_dir:
                try:
                    inference_tgnet(lower_scan, upper_scan, output_dir)
                    response = {"status": "success", "message": "Inference completed successfully."}
                except Exception as e:
                    response = {"status": "error", "message": str(e)}
            else:
                response = {"status": "error", "message": "Invalid input or output directory."}

            await websocket.send(json.dumps(response))
    except websockets.ConnectionClosed:
        print("Connection closed")
    except Exception as e:
        print(f"Unexpected error: {e}")

async def main():
    async with websockets.serve(handle_connection, "localhost", 8800):
        print("WebSocket server started on ws://localhost:8800")
        await asyncio.Future()  # Run forever

# Use asyncio.run for compatibility with Python 3.7+
if __name__ == "__main__":
    asyncio.run(main())