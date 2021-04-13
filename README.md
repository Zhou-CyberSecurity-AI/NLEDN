# NLEDN
Image rain mitigation model

# Abstract
Single image rain streaks removal has recently witnessed substantial progress due to the development of deep convolutional neural networks. However, existing deep learning based methods either focus on the entrance and exit of the network by decomposing the input image into high and low frequency information and employing residual learning to reduce the mapping range, or focus on the introduction of cascaded learning scheme to decompose the task of rain streaks removal into multi-stages. These methods treat the convolutional neural network as an encapsulated end-to-end mapping module without deepening into the rationality and superiority of neural network design. In this paper, we delve into an effective end-to-end neural network structure for stronger feature expression and spatial correlation learning. Specifically, we propose a non-locally  enhanced encoder-decoder network framework, which consists of a pooling indices embedded encoder-decoder network to efficiently learn increasingly abstract feature representation for more accurate rain streaks modeling while perfectly preserving the image detail. The proposed encoder-decoder framework is composed of a series of non-locally enhanced dense blocks that are designed to not only fully exploit hierarchical features from all the convolutional layers but also well capture the long-distance dependencies and structural information. Extensive experiments on synthetic and real datasets demonstrate that the proposed method can effectively remove rain streaks on rainy image of various densities while well preserving the image details, which achieves significant improvements over the recent state-of-the-art methods.

![image](https://user-images.githubusercontent.com/35444743/114492579-6d969f00-9c4b-11eb-81be-8f823ba59ef0.png)
Figure: The overall architecture of our proposed non-locally enhanced encoder-decoder network (NLEDN). As can be observed, the input image and low-level feature activation are linked to the very end of the whole architecture via long-range skip-connections. The core of the whole architecture is a non-locally enhanced encoder-decoder, in which novel non-locally enhanced dense blocks (NEDBs) and pooling indices guided scheme are adopted.

![image](https://user-images.githubusercontent.com/35444743/114492730-bc443900-9c4b-11eb-85a9-2594e3d6a864.png)
Figure 3: The architecture of our proposed non-locally enhanced dense block (NEDB). Left part shows the multi-scale input via either adopting global-level non-local enhancement which feeds the entire feature map to NEDB or dividing the feature map into a grid of regions to realize region-level non-local enhancement. Here we show by a 2 × 2 grid for convenience.

![image](https://user-images.githubusercontent.com/35444743/114492846-f7466c80-9c4b-11eb-9fb8-0063e555233a.png)
![image](https://user-images.githubusercontent.com/35444743/114492851-fe6d7a80-9c4b-11eb-8110-40b6361891e5.png)
![image](https://user-images.githubusercontent.com/35444743/114493194-ae42e800-9c4c-11eb-8a5a-5f9681cc87fd.png)

# Dataset Dowload:
link：https://pan.baidu.com/s/1vm5iqt-ONycAdh9qXLJg_A 
code：uxq8 


# Running Steps
1.Download four testing sets (Rain100L, Rain100H, DDN, DID-MDN).

2.Choose the type of dataset in Congig.config.py.
  
3.Run train.py. to train the NLEDN model

4.Run test.py. to test the trained model
