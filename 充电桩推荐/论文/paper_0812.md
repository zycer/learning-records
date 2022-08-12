本文研究了基于墨卡托投影与交互式投票的地图匹配算法，将GPS设备采集到的位置数据匹配到正确的道路上。我们的方法首先使用卡尔曼滤波法纠正GPS数据中偏差过大的数据点，然后基于GPS采样点之间的相互影响，分别采用空间分析、时间分析以及道路分析来度量连续候选点之间的关系，通过引入三种月数条件，进一步过滤掉数据中的噪声，提升了算法的效率，使用墨卡托投影算法以及球面距离的计算提高了算法的准确度。此外，MIVMM算法使用候选边缘投票，该投票方法相较于候选点的投票对连续点之间的关系更加敏感，因此进一步提高了投票的准确率。通过上一节的实验表明，MIVMM算法对AIVMM算法做出了改进，尤其对于路网结构复杂的情况下大幅提高了算法的准确率，且通过优化查找局部最优路径算法以及使用了启发式star*算法查缩短了寻找最短路径所消耗的时间，使用全局KNN搜索候选路段降低了匹配非最近路段的可能性，使得MIVMM算法在三种路段上的准确率都达到了90%以上。

Study of GPS Data De-noising Method Based on Wavelet and Kalman filtering
A Novel Hybrid Fusion Algorithm for Low-Cost GPS/INS Integrated Navigation System During GPS Outages
Low-cost IMU Data Denoising using Savitzky-Golay Filters
A GPS Positioning Algorithm Based on Distributed Kalman Filter Data Fusion with Feedback *
Producing the location information with the Kalman filter on the GPS data for autonomous vehicles
