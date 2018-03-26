# jazz-show-case

This script loads several instrument recognition neural networks and makes predictions on a collection of jazz solos. The instruments are: alto saxophone (as), tenor saxophone (ts), soprano saxophone (ss), trombone (tb), trumpet (tp), and clarinet (cl). You can chose between loading a model that uses the mixed audio or the solo separated audio. The neural network is based on the work by Han et al. (Deep convolutional neural networks for predominant instrument recognition - 2016) and the solo/accompaniment separation uses the algorithm by Cano et al. (Pitch-informed solo and accompaniment separation towards its use in music education applications - 2014).

## Prerequisites

```
pip install keras, matplotlib, numpy, librosa, h5py
```

## Predict and Plot

Just run the main file.

```
python3 main.py
```

![alt text](https://github.com/juansgomez87/jazz-show-case/blob/master/jazz_show_case.png)
