# UltraCEGAN: Ultrasound Image Generation using CycleGAN

UltraCEGAN is an open-source project aimed at generating ultrasound images that resemble contrast-enhanced ultrasound imaging using the CycleGAN framework. By leveraging the power of deep learning and generative adversarial networks (GANs), this project provides a novel approach to synthesizing ultrasound images with enhanced contrast.

## Features

- **CycleGAN-based:** UltraCEGAN utilizes the CycleGAN architecture, which enables the translation between two domains - in this case, regular ultrasound images and contrast-enhanced ultrasound images.
- **Realistic Image Generation:** The trained UltraCEGAN model can generate ultrasound images that closely resemble the appearance and characteristics of contrast-enhanced ultrasound imaging.
- **Medical Imaging Applications:** UltraCEGAN has potential applications in medical imaging research, such as aiding in the development of new imaging techniques, enhancing training data for machine learning algorithms, and facilitating the evaluation of image reconstruction algorithms.

## Getting Started

To get started with UltraCEGAN, follow the instructions below:

1. Clone the UltraCEGAN repository: `git clone https://github.com/XDURG/UltraCEGAN.git`
2. Install the required dependencies listed in the `requirements.txt` file.
3. Obtain the pre-trained UltraCEGAN model from us upon reasonable request or train your own model using your own dataset using `cyclegan_change_losscalc.py`.
4. Run the inference script to generate ultrasound images.

## Contributing

Contributions to UltraCEGAN are welcome! If you find any issues, have suggestions for improvements, or would like to add new features, feel free to open an issue or submit a pull request.

## License

UltraCEGAN is released under the [Apache License 2.0](LICENSE).
