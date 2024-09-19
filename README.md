# CREATING TACTILE PHOTOS THAT PEOPLE WITH VISUAL IMPAIRMENTS CAN ENJOY USING GENERATIVE AI AND 3D PRINTERS

Thesis files are available [here](https://github.com/elsoulevor/Thesis-files)

## How to achieve the results obtained (not hard)

### Repositories used
We tried using 2 different depth estimation models to generate supposedly accurate depth maps, that are then to be converted to 3D meshes.

[Self-Reference Distillation](https://github.com/rnlee1998/SRD-Depth)

Please follow the setup instrucitons thoroughly. We don't need to install the dataset, just the requirements. Create a virtual environment first, and then in the 'requirements.txt' file, remove any line that is related to a local repository.

[Marigold](https://github.com/prs-eth/marigold?tab=readme-ov-file)

Same for the Marigold model.

The Marigold model proves very efficient, and s the model we will be fixed on.

### Generating 3D models

Head to the /pipeline repository, and to test the 3D model generation on a given depth map, run the command following this example
```shell
python3 pipeline.py --colormap Spectral --image_path */pipeline/tests/example_1_pred_colored.png
```

Make sure to check what spectral map is used when generating the depth map so that you input the right one in the command.

Modify as you please to get the output in the desired repository.

## Future work

### Estimation on segmented objects
As you may have noticed, there is an almost empty <i>focus_segmentation</i> repository. If you have read the thesis, we were exploring applying the depth estimation on each individual object so that it can possibly solve
the flatness problem. This is method is to be further developed.

### Sky segmentation
Some methods exist to remove the sky. We were thinking of creating a method that can detect if the sky is a relevant element of the image to decide to keep it or not. If it is not, it can add bias to the depth map.
This is to be further explored.

### Evaluation method

Since there is not any work similar to what we are doing here, we need a way to evaluate our results.
