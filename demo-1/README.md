# Image Search Demo

## Requirements

* streamlit 0.84.1 or later
* nmslib 2.1.1 or later

In addition, we need to generate CLIP vectors for the image corpus using `demo-image-encoder.ipynb`.

## Running

* Run the `run_demo_1.sh` script in the `demo-1` folder, this will start up a server listening on local port 8501.
* Open a SSL tunnel to your localhost on port 8501, using command: `gcloud alpha compute tpus tpu-vm ssh fishtail --zone us-central1-a --project hf-flax --ssh-flag="-L 8501:localhost:8501"`
* Start the application on your browser at `http://localhost:8501`.


