### Results
**Here shows how the GIFs distribute after clustering. We can see that GIFs with different visual effects distribut at different spaces.**

![image](https://github.com/zzzzdh/Seenomaly/blob/master/image/cluster.gif)

### Given a linted GUI, We can use KNN to find the most possible category in this space.

**Our model does not simply classify different animation. It can actually distinguish GUIs with the same type of animation but different visual effects, which is demonstrated in this figure:**

![image](https://github.com/zzzzdh/Seenomaly/blob/master/image/d.png)

The two red circles show the same type of animation (pop-up window) but with different visual effects. One is with black background or shadow (1), which means that the pop-up  window can be distinguished clearly from the background and thus they are treated as normal examples. The other one contains pop-up windows over a non-shadow background (2), and thus are treated as violation examples.

The two green circles show another example - the animation of bottom sheet over a background with/without shadow. Samples with shadow or over black background (3) are treated as normal examples. But those without shadow or black background (4) violate the Material-Design don't guideline regarding the shadow, and thus are treated as violation examples.

This figure illustrates that our model is able to to distinguish normal and violation GUI animations in the feature space. Note that this visualization is only a 2-dimensional visualization of a 2048-dimensional feature space. In this feature space, if a to-be-linited GUI animation has certain violation, it will be place close to the similar animation vioaltion examples. As a result, k-nearest-neighbors vote can idenitfy the likely violation in the to-be-linited animation.
