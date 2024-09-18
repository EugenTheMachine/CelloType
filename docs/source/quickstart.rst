Quickstart
------------------------------

Clone the repository:

.. code-block:: console

    git clone https://github.com/maxpmx/CelloType.git
    cd CelloType

Then Download the model weights:

.. code-block:: console

    cd data
    sh download.sh
    cd ..

Predict on an example image:

.. code-block:: python

    from skimage import io
    from cellotype.predict import CelloTypePredictor

    img = io.imread('data/example/example_tissuenet.png') # [H, W, 3]

    model = CelloTypePredictor(model_path='./models/tissuenet_model_0019999.pth',
      confidence_thresh=0.3, 
      max_det=1000, 
      device='cuda', 
      config_path='./configs/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml')

    mask = model.predict(img) # [H, W]