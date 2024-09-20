import asyncio
import os
import pickle
import time
import wave
import numpy as np
import sounddevice as sd  # Import sounddevice for audio playback
import queue  # Import queue for thread-safe communication between threads
import traceback
import logging
import json

logger = logging.getLogger(__name__)
# Debugging logger
logging.basicConfig(level=logging.INFO)

# For the sake of this example, we'll define dummy length_prefixing functions
def length_prefixing(response):
    serialized_response = pickle.dumps(response)
    response_length = len(serialized_response)
    response_header = response_length.to_bytes(4, 'big')
    return response_header + serialized_response

async def recv_with_length_prefixing(reader):
    header = await reader.read(4)
    if not header:
        return header
    # recv with specified length
    res_length = int.from_bytes(header, 'big')
    data = bytearray()
    while len(data) < res_length:
        res = await reader.read(res_length - len(data))
        if not res:
            raise ConnectionError("Socket connection lost")
        data.extend(res)
    
    return data

async def record_track2(MainServerA_config, MainServerB_config, promptA, promptB, output_dir, output_name):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Hosts and ports for MainServerA and MainServerB
    HOST_A = MainServerA_config["host"]
    HOST_B = MainServerB_config["host"]
    PORT_A = MainServerA_config["port"]
    PORT_B = MainServerB_config["port"]

    # Connect to the servers
    reader_A, writer_A = await asyncio.open_connection(HOST_A, PORT_A)
    reader_B, writer_B = await asyncio.open_connection(HOST_B, PORT_B)

    # Create bridge tasks to connect the servers
    bridge_a_to_b_task = asyncio.create_task(
        bridge_task(reader_A, writer_B, output_dir, output_name + '_A.wav', 'A_to_B'))
    bridge_b_to_a_task = asyncio.create_task(
        bridge_task(reader_B, writer_A, output_dir, output_name + '_B.wav', 'B_to_A'))

    try:
        # Set system prompts
        writer_A.write(length_prefixing({"type": "system_prompt", "data": promptA, "input_timestamp": time.time()}))
        await writer_A.drain()

        writer_B.write(length_prefixing({"type": "system_prompt", "data": promptB, "input_timestamp": time.time()}))
        await writer_B.drain()

        # After setting system prompts
        # writer_A.write(length_prefixing({"type": "user_text", "data": "Hello", "input_timestamp": time.time()}))
        # await writer_A.drain()
        
        # Run all tasks concurrently
        await asyncio.gather(
            bridge_a_to_b_task,
            bridge_b_to_a_task,
        )
    except KeyboardInterrupt:
        print("Shutting down servers...")
    finally:
        # Cancel all tasks
        bridge_a_to_b_task.cancel()
        bridge_b_to_a_task.cancel()
        writer_A.close()
        writer_B.close()
        await writer_A.wait_closed()
        await writer_B.wait_closed()

async def bridge_task(reader, writer, output_dir, output_filename, bridge_name):
    """
    Receives audio data from one server, plays it, records corresponding frames, and sends it to the other server.
    """
    # Set up WAV file recording
    wav_file_path = os.path.join(output_dir, output_filename)
    wav_file = wave.open(wav_file_path, 'wb')
    wav_file.setnchannels(1)  # Mono audio
    wav_file.setsampwidth(2)  # 16-bit audio
    wav_file.setframerate(16000)  # Sample rate

    # Set up JSON Lines file for non-audio data
    json_file_path = os.path.join(output_dir, output_filename.replace('.wav', '.jsonl'))
    json_file = open(json_file_path, 'a')  # Open in append mode

    print(f"Bridge {bridge_name} started. Outputting audio to {wav_file_path}...")

    SAMPLE_RATE = 16000
    CHANNELS = 1

    # Create thread-safe queues for audio data
    output_queue = queue.Queue()  # For data to play
    input_queue = queue.Queue()   # For data to send to the other server
    wav_queue = queue.Queue()     # For data to write to wav file

    # Define the audio callback function
    def audio_callback(outdata, frames, time_info, status):
        if status:
            print(f"Status: {status}", flush=True)
        try:
            # Use leftover data if available
            if hasattr(audio_callback, 'leftover'):
                data = audio_callback.leftover
                del audio_callback.leftover
            else:
                try:
                    data = output_queue.get_nowait()
                except queue.Empty:
                    data = np.zeros(frames, dtype='int16')
        except IndexError:
            data = np.zeros(frames, dtype='int16')
        # Ensure data is the correct length
        if len(data) < frames:
            # Pad data with zeros if not enough data
            data = np.pad(data, (0, frames - len(data)), 'constant')
        elif len(data) > frames:
            # Save extra data to be used first in the next callback
            audio_callback.leftover = data[frames:]
            data = data[:frames]
        outdata[:] = data.reshape(-1, CHANNELS)

        # Put the output data into wav_queue to write to wav file
        wav_queue.put(outdata.copy())
        input_queue.put(outdata.copy())

    # Open the sounddevice Stream in output mode
    with sd.OutputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16',
                     callback=audio_callback):
        try:
            # Create async tasks to send data to the other server and write to wav file
            async def send_to_server():
                loop = asyncio.get_event_loop()
                while True:
                    # Get data from the input queue in a thread-safe way
                    input_data = await loop.run_in_executor(None, input_queue.get)
                    # Prepare the data to send
                    audio_res = {
                        "type": "audio",
                        "data": input_data,
                        "input_timestamp": time.time()
                    }
                    data_to_send = length_prefixing(audio_res)
                    logging.debug(f"Sending audio data from {bridge_name}: {input_data}")
                    writer.write(data_to_send)
                    await writer.drain()

            async def write_wav_file():
                loop = asyncio.get_event_loop()
                while True:
                    data = await loop.run_in_executor(None, wav_queue.get)
                    wav_file.writeframes(data.tobytes())

            send_task = asyncio.create_task(send_to_server())
            write_wav_task = asyncio.create_task(write_wav_file())

            while True:
                logging.debug(f"Waiting for data from {bridge_name}...")
                # Receive data from server
                res = await recv_with_length_prefixing(reader)
                if not res:
                    break

                res = pickle.loads(res)
                # Check if it's system audio data
                if res["type"] == "system_audio":
                    audio_data = res["data"]
                    if audio_data is not None:
                        # Put the audio data into the output queue
                        logging.debug(f"Received audio data from {bridge_name}: {audio_data}")
                        output_queue.put(audio_data)
                    else:
                        # Handle end of audio if necessary
                        pass
                else:
                    # Write non-audio data to JSON Lines file
                    json_line = json.dumps(res)
                    json_file.write(json_line + '\n')
                    logging.info(f"Written non-audio data to JSON file from {bridge_name}: {res}")

                # Handle other types if necessary
                await asyncio.sleep(0)  # Yield control to the event loop
        except Exception as e:
            print(f"Exception in bridge_task {bridge_name}: {e}")
            traceback.print_exc()
        finally:
            # Close the WAV and JSON files
            send_task.cancel()
            write_wav_task.cancel()
            try:
                await send_task
            except asyncio.CancelledError:
                pass
            try:
                await write_wav_task
            except asyncio.CancelledError:
                pass
            wav_file.close()
            json_file.close()
            print(f"Audio output saved to {wav_file_path}")
            print(f"Non-audio data saved to {json_file_path}")

if __name__ == "__main__":
    # Example usage:
    MainServerA_config = {"host": "140.112.21.20", "port": 43007}
    MainServerB_config = {"host": "140.112.21.20", "port": 43008}
    promptA = "You are a client planning a vacation to Kos Island, Greece. You are now talking to a travel agent, ask some questions to finalize the trip."
    promptB = "You are a travel agent helping a client plan a vacation to Kos Island, Greece. You offer destination options, packages, and recommendations based on the client's needs."
    output_dir = "./output"
    output_name = "conversation"

    asyncio.run(record_track2(MainServerA_config, MainServerB_config, promptA, promptB, output_dir, output_name))
