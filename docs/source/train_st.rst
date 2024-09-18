Cell Segmentation (Xenium Spatial Transcriptomics)
------------------------------

Download data and pretrained models weights
~~~~~~~~~~~~~~~~~~~~~~~~~

Download the processed data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**IMPORTANT**: Note that the raw data is from `Xenium Human Lung Dataset <https://www.10xgenomics.com/datasets/preview-data-ffpe-human-lung-cancer-with-xenium-multimodal-cell-segmentation-1-standard>`_. This processed data is for demo purpose ONLY!

Download ``data/example_xenium.zip`` from the `Drive <https://upenn.box.com/s/str98paa7p40ns32mchhjsc4ra92pumv>`_ and put it in the ``data`` folder. Then unzip it.

.. code-block:: bash

    cd data
    unzip example_xenium.zip
    cd ..

Download COCO pretrained models weights (optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Download ``models/maskdino_swinl_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask52.3ap_box59.0ap.pth`` from the `Drive <https://upenn.box.com/s/str98paa7p40ns32mchhjsc4ra92pumv>`_ and put it in the ``cellotype/models`` folder.

Train model
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    python train_xenium.py --num-gpus 4

The parameters are optimized for 4\*A100 (40GB) environment, if your machine does not have enough GPU memory, you can reduce the batch size by changing the ``IMS_PER_BATCH`` in ``configs/Base-COCO-InstanceSegmentation.yaml``. For reference, the training take ~10 hours on 4\*A100 (40GB) environment.

Test model and visualize results
~~~~~~~~~~~~~~~~~~~~~~~~~

For reference, our trained weights ``models/xenium_model_0001499.pth`` can be downloaded from the `Drive <https://upenn.box.com/s/str98paa7p40ns32mchhjsc4ra92pumv>`_ folder.

.. code-block:: bash

    python test_xenium.py --num-gpus 1

The example prediction saved in the ``output/xenium`` folder.

.. image:: output/xenium/0_pred.png
    :width: 150px
    :alt: drawing