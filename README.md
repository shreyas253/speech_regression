# speech_regression
 
INTRODUCTION
------------
The following is forms example code with random data for that performs trains regression models using Standard GMMs, Bayesian GMMs and feed-forward DNNs and models adaptation for the paper:
 
Shreyas Seshadri, Lauri Juvela, Okko Räsänen and Paavo Alku: "Vocal Effort Based Speaking Style Conversion Using Vocoder Features and Parallel Learning", IEEE Access, vol. 7, pp. 17 230–17 246, 2019. [OpenAccessLink](https://ieeexplore.ieee.org/abstract/document/8631106)

Please download the [Voicebox]( http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html) and [GMMbayes]( http://www.it.lut.fi/project/gmmbayes/downloads/src/gmmbayestb/) toolboxes and to the [./Mapping/SGMM/](https://github.com/shreyas253/speech_regression/tree/master/Mapping/SGMM) folder
 
Comments/questions are welcome! Please contact: shreyas.seshadri@aalto.fi
 
Last updated: 8.8.2018
 
 
LICENSE
-------
 
Copyright (C) 2018 Shreyas Seshadri Aalto University
 
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:
 
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
 
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 
The source code must be referenced when used in a published work.
 
METHODS
-------
SGMMs - The standard Expectation-maximization (EM) algorithm- based Gaussian mixture models (SGMMs) that is used for regression in speech applications [1]. Model adaptation is done similar to speaker verification research [2].
BGMMs - The Bayesian extension to SGMMs. See [3] and [4] for reference. See [code](https://github.com/shreyas253/BGMM_Mapping) for original implementation. Model adaptation is simply handled by using the pre-trained model as prior for the new data. 
DNNs - Feedforward DNNs (see [5]). For Model adaptation the DNN is trained normally with the new data but the weights initialized from the pre-trained DNN. 
 
 
REFERENCES
----------
[1] A. Kain and M. W. Macon. Spectral voice conversion for text-to-speech synthesis, in Proc. ICASSP, Seattle, USA, 1998, pp. 285-288.
 
[2] D. A. Reynolds, T. F. Quatieri, and R. B. Dunn. Speaker verification using adapted Gaussian mixture models, Digital signal processing, vol. 10, no. 1-3, pp. 19-41
 
[3] D. M. Blei and M. I. Jordan. Variational inference for Dirichlet process mixtures, Bayesian analysis, vol. 1, no. 1, pp. 121-144, 2006.
 
[4] C. M. Bishop, Pattern recognition and machine learning. Springer, 2006.
 
[5] I. Goodfellow, Y. Bengio, and A. Courville, Deep Learning. MIT Press, 2016, http://www.deeplearningbook.org.
