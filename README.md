
[//]: # (Image References)

[image1]: ./myapp2/static/sample/chihuahua_output.jpg "Sample Output Dog"
[image2]: ./myapp2/static/sample/mysample.jpg "Sample Output Human"


## Dog Classification Web App Project

Goal is to build a pipeline that can be used within a web or mobile app to process real-world, user-supplied images. In this specific workload, the algorithm will be approximating the breed of a canine within a given image. If supplied an image of a human, the code will identify the resembling dog breed.  

__Dec. 10, 2020 Update__:
* The app is deployed and available for user inputs. Please feel free to try the app via https://jiae.ai


![Sample Output][image1]
![Sample Output][image2]

    Top image is my Chihuahua. My model has never seen the image, and yet, even with the image slightly squished, it matched the breed!
    
    Bottom is me, in which the model detects my face as human and matches my appearance with a dog breed, Lowchen!

By utilizing the techniques, including Transfer Learning, building a Convolutional Neural Network(CNN), auto-encoders and object detection, this project will explore the possibilities of CNN models in classification and localization, as well as engineering different models together to yield optimal results in specified tasks and user-experiences.

## References
* [dog image dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip). 		
	- The unzipped, `dogImages/` folder should contain 133 folders, each corresponding to a different dog breed.
* [human face dataset](http://vis-www.cs.umass.edu/lfw/lfw.tgz).  
	- [7zip](http://www.7-zip.org/) is recommended for extracting the folder, if Windows machine is used.
