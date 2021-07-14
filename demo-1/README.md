# Image Search Demo

## Requirements

* streamlit 0.84.1 or later
* nmslib 2.1.1 or later

In addition, we need to generate CLIP vectors for the image corpus using `demo-image-encoder.ipynb`.

## Running

* Run the `run_demo_1.sh` script in the `demo-1` folder, this will start up a server listening by default on local port 8501. In case someone is using that port already, streamlit will assign the next available port, you should see it logged on the terminal where you started the script.
* Open a SSL tunnel to your localhost on port 8501 (or whatever port is reported by the script), using command: `gcloud alpha compute tpus tpu-vm ssh fishtail --zone us-central1-a --project hf-flax --ssh-flag="-L 8501:localhost:8501". This forwards port 8501 on the gcloud instance to port 8501 on your local machine. If the port reported is different or you want it forwarded to a different port on your machine, make the appropriate modifications to your command.
* Start the application on your browser at `http://localhost:8501` (or whatever port you forwarded to).


