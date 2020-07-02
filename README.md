# Daytime_Arctic_Cloud_Detection_Practice

* This project mainly explores and analyzes the classification methods in relevance to the paper Daytime Arctic Cloud Detection Based on Multi-Angle Satellite Data With Case Study written by Shi et al.
* Latex files and figures all contained in folder 'report'
* All code used in this project contained in folder 'code', including R code and Python code to perform random forest algorithm
* CVgeneric code is contained in folder 'code' as 'CVgeneric.R'

## Conclusions

* The second splitting method \textbf{Image Based Method} is of more practical meaning than the first method, though  with lower accuracy, since it is applied to data by picture as a whole instead of by picked pixel grid and thus more applicable for future data.
* Under the second splitting method, the Random forest classification perform very well in some specific regions where all other methods fail, and though its accuracy is not the highest (which is obtained by QDA method), the errors mainly occur near boundaries which can be smoothed and thus the accuracy would increase.
* The optimization method applied to random forest classification is to increase its accuracy based on the expert labels. But the smoothing may be not that pragmatic since the experts tend to give labels to entire regions while in fact there are always cloud and not cloud small areas in an entire region.

## Contributing

Please read the last part of our report, which included in folder 'report',  for details on our code of conduct, and the process for submitting pull requests to us.

## Authors

* **Xiaoyang Wu** 
* **Lanxin Zhang**  

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Thanks GSIs for answering our questions
* Thanks Google for a lot of information
* Thanks the paper provided, it offers a lot of inspiration for future statistic studying
