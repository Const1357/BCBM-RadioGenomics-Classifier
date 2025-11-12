# BCBM-RadioGenomics-Classifier

IMPORTANT: Before running the code, activate the virtual environment by running:

```source BCBM_classifier_env/bin/activate```

source BCBM_classifier_env/bin/activate             

# Enable SSH connection (if pc shuts down)

tailscale up

tailscale ip -4

verify tailscale ip is 100.99.202.57 (usually stays the same)

OTHERWISE: (on laptop)
change ssh configs to match new ip


enable tensorboard with: tensorboard --logdir models/
then open http://localhost:6006 in browser


2 4 7 models for ensemble