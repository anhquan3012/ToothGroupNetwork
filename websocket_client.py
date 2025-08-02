# websocket_client.py
import asyncio
import websockets
import json

async def test_inference():
    uri = "ws://localhost:8800"
    async with websockets.connect(uri) as websocket:
        # Replace these paths with your actual input and output directories
        lower_scan = "C:\\Users\\Quan Tran\\ToothGroupNetwork\\samples\\SAMPLE1\\SAMPLE1_lower.stl"
        upper_scan = "C:\\Users\\Quan Tran\\ToothGroupNetwork\\samples\\SAMPLE1\\SAMPLE1_upper.stl"
        output_dir = "C:\\Users\\Quan Tran\\ToothGroupNetwork\\results"

        request = {
            "lower_scan": lower_scan,
            "upper_scan": upper_scan,
            "output_dir": output_dir
        }

        await websocket.send(json.dumps(request))
        response = await websocket.recv()
        print(f"Response from server: {response}")

# Use asyncio.run for compatibility with Python 3.7+ and better async handling
if __name__ == "__main__":
    asyncio.run(test_inference())
