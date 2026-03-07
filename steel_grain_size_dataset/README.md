# About this directory

Suggest Edits
AG = Artificial Grain
AGMask = Artificial Grain Original Voronoi Pattern (no noise)

RG = Real Grains
RGMask = Manually Segmented Grains (done in paint)

HED_PRE = Holistically nested edge detection method
GRAD_PRE = Gradient Based Approach Method
THRESH_PRE = Threshold approach

# Description
This is a dataset that is of grain boundaries. The material is stainless steel, and it was produced through additive manufacturing. It is Stainless Steel 316L printed on an ExONE printer. This is a form of Binder Jetting Additive Manufacturing.

2.25 pixels is one micron

The images were taken on an optical microscope at 500X. I then split the 21 collected images (1600X1200) into 12 images at (400X300).

To produce the grain boundaries I just used traditional image thresholding methods. Please check out my youtube channel for tutorials on some of these methods. Here is an example of one: https://www.youtube.com/watch?v=od5G39W7eYc&t=612s. The folder RG is the real grains, the folder RG mask is a manually segmented (in paint) that can be used to train a CNN.

There are a lot of good code examples on kaggle of image segmentation methods as well.

I put this code on here to train a U-Net to detect the grain boundaries. The ultimate goal is to get a lot of grain datasets on here and get a U-NET that can detect and segment the grain boundaries. A universal deep learning grain boundary segmentation method.

There is also a AG file which is artificial grain. This was generated with a Voronoi Cell generation method. Check out chapter 4 of my dissertation for more details on how this is done!!

https://stars.library.ucf.edu/etd2020/1693/

and look at this old dataset for some different code ideas , I used some different methods here:
https://www.kaggle.com/datasets/peterwarren/exone-stainless-steel-316l-grains-500x
