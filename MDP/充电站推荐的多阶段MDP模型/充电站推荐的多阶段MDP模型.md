# 摘要
为了最大限度地缩短车辆的总充电时间和平衡充电站的负载，越来越需要充电站推荐。为了满足这一需求，我们将推荐问题建模为马尔可夫决策过程(MDP)问题。然而，传统的MDP模型存在“维度诅咒”的问题。针对这一问题，我们提出了MDP的一种扩展：多阶段MDP，将MDP的状态转换分解为多个阶段，以减少状态空间和状态转换的复杂度。这是通过引入MDP中定义的正常状态以外的两种状态来实现的：后决策状态和中间决策状态。在此基础上，提出了一种基于在线学习的多阶段MDP模型求解算法。由于减少了状态空间和状态转移的复杂度，该在线算法能够快速收敛。通过与基于博弈论的推荐机制和基于Q-学习的推荐机制的比较，仿真结果表明本文提出的推荐方案具有良好的性能。
# 1. 引言
电动汽车的普及率不断提高，使得电动汽车充电成为一个关键问题。安装在家里的充电桩只能解决部分问题，由于充电时间长，车辆无法到达，如电动出租车、Uber电动汽车。因此，越来越多的支持电动汽车加油的公共基础设施开始实施并商业化。然而，随着电动汽车普及率的不断提高，充电需求与现有充电基础设施容量之间的差距始终存在。
​

已经确定了与充电问题相关的几个挑战，包括充电事件的路由优化[1]、范围估计[2]、充电站位置[3]、充电调度[4]和充电站（CS）建议[5]等。在这些挑战中，充电调度和充电建议是两个重要的挑战。充电计划是为电动汽车安排合适的充电时间段，而充电建议是在电动汽车请求充电时向其推荐最佳CS。这两个挑战在某种程度上是相似的，因此在文献中，充电调度被分类为时间调度，而充电建议被分类为空间调度[6]。
​

充电调度的主要目标是减轻电动汽车对电网的影响，同时保持可接受的用户满意度，而CS建议的目标是最小化包括驾驶时间、排队时间和充电时间在内的总充电时间。从用户的角度来看，收费的体验质量（QoE）是主要关注点。因此，推荐具有最佳CS的电动汽车以通过最小化总充电时间来改善用户QoE至关重要。此外，CS建议或多或少会影响其他问题。例如，它影响电动汽车的布线优化[1]和CS布局[7]。
​

为了提高CS推荐的性能，目前的工作主要集中在两个方面。一个是设计一个建议架构/方案，帮助电动汽车选择CS并完成充电[8]，[9]，另一个是建议建模。在我们的工作中，我们重点研究了推荐模型，它对决定CS推荐的性能起着至关重要的作用。
​

由于CS推荐可以看作是一个分配问题（将一个EV分配给一个CS），因此相关的建模方法可以用于CS推荐。事实上，智能交通系统的许多问题，如出租车调度和停车诱导等，也是一个分配问题。因此，在这些领域提出的模型/算法可以纳入CS建议，如出租车调度的多重优化模型[10]、[11]和停车诱导的二次分配问题模型[12]。目前，动态建模引起了该领域的广泛关注，如基于博弈论的建模[6]、[13]、动态资源分配[14]等。
​

尽管这些工作对CS推荐模型的建模做出了很大的贡献，但在建模复杂度和效率方面仍存在一些不足。静态优化模型，如[15]中使用的线性规划(LP)和[16]中使用的混合整数规划(IP)，可以实现全局（或近全局）优化（例如，最小化收费时间或最大化收入），但不适用于推荐场景。原因之一是这些建模不能捕捉CS建议的动态特性。二是它们通常需要大量的计算，因此会产生大量的延迟。动态建模，如[13]中使用的博弈论，确实反映了推荐的动态特征。然而，以往的动态建模要么实现了一步优化，而没有对未来的预测，要么无法进行智能推荐。这促使我们设计一个基于扩展马尔可夫决策过程(MDP)模型的动态智能CS推荐系统。
​

在我们的CS推荐系统中，当收到电动汽车的充电请求时，系统需要决定电动汽车应该被引导到哪个CS。该决策是基于系统的当前状态做出的，其中包括例如CSS的排队信息和距离信息。一个最优决策必然会使总的充电时间最小化，总的充电时间包括开车时间、排队时间和充电时间。对于每一个决定，我们让系统获得一个与总收费时间成反比的奖励。同时，系统演化到下一个状态。因此，整个过程可以被建模为一个MDP问题，其中的关键成分是一组决策环境、系统状态、可用行动、奖励和依赖于状态/行动的转移概率[17]。
​

根据以上分析，MDP模型很好地抓住了CS推荐的特点。然而，MDP有“维度诅咒”[18]的问题，在CS推荐方案中，这一问题变得更加严重，因为存在较大的状态/动作空间。此外，由于MDP建模需要对模型的所有概率分布有先验知识，这在实际中是不可能的。为了解决这些问题，我们引入了决策后状态(PDS)[19]，并在模型中引入了一种新的状态，称为中间决策状态(IDS)，我们称之为多阶段MDP(MMDP)模型。利用这两个附加状态，将MDP的状态转换分解为多个阶段，从而大大降低了状态空间和状态转换的复杂性。在此基础上，提出了一种基于在线学习的MMDP模型求解算法。
​

本文的主要贡献如下：

- 提出一种新的MDP模型——MMDP。在该模型中，除了MDP中定义的正常状态外，还引入了两种附加状态PDS和IDS。
- 将CS推荐问题描述为一个MMDP模型。与传统的MDP模型相比，MMDP在状态空间和状态转移复杂度方面都有明显的降低。这使得我们的命题对于存在“维数诅咒”问题的CS建议是有效的。
- 提出了一种基于在线学习的MMDP模型求解算法。理论分析和仿真评估都表明，基于在线学习的算法收敛速度快，在总体充电时间和CSs负载均衡方面具有良好的充电性能。



本文的其余部分组织如下。第二节简要总结了相关工作。第三节描述了使用MDP的电动汽车推荐问题。第四节通过介绍PDS和IDS来阐述我们的MMDP模型。第五节介绍了求解MMDP模型的在线学习算法。第六节分析了本文命题的趋同性。第七节通过仿真对性能进行了评价。第八节总结了本文的工作并提出了展望。
# 2. 相关工作
CS布局（部署）影响调度和再指挥性能，已被广泛研究。可以充分利用现有加油站进行CS建设[20]，也可以寻求新的CS设置方案[21]。CS布局问题可以看作是一个在充电需求、续航里程等多个约束条件下的优化问题，并被断言为NP完全问题[3]、[21]。为了解决这个优化问题，Albert Y.S.林等人。[3]提出了迭代混合整数线性规划(MILP)和贪婪法等解决方案，以最小化建设成本。[21]设计了两个基于贪婪的多目标同时优化算法。CS的部署也可以被认为是一个博弈问题，Luo等人。[22]将其表述为一个贝叶斯博弈，旨在实现CS所有者利润、用户满意度和电网可靠性之间的平衡。
​

充电调度解决了电动汽车在CS或公园停车接受检查时应该何时充电的问题。适当的电动汽车充电调度可以维持电网的稳定，保证电力供需平衡[23]。Mukherjee和Gupta[4]将这些工作总结为单向充电和双向充电两大类。前者是指电网到车辆(G2V)，电动汽车只从电网充电；后者是指车辆到电网(V2G)，电力流动是双向的。
​

将以往的调度工作分为车站/停车级调度和网格级调度两大类。对于车站/停车层，调度在每个CS或停车处单独执行。在[24]中，为了使平均排队长度（即充电时间）最小化，一个CS确定排队中待服务的电动汽车的数量和待分配用于充电的可再生能源的数量。作者将其建模为约束MDP问题。在文献[25]中，作者设计了一个最优定价方案来指导和协调电动汽车在CS中的充电过程。该问题被建模为一个凸优化问题，目标是最小化服务丢失率。在[26]中，作者将调度问题建模为多模式近似动态规划(MM-ADP)，目标是最小化收费费用。在[27]中讨论了停车库下的调度问题。在实现运营商利润最大化的同时，为计费用户提供满意的服务。
​

网格级调度的目标是实现全局优化，而不是单个控制系统的优化。在文献[28]中，作者在考虑电网约束的情况下，设计了一个使发电机运行费用最小的最优计费控制计划。另一方面，收费成本最小化被认为是一个目标[29]，[30]。文[30]将充电调度问题描述为一个约束MDP问题，其目的是在保证电动汽车充满电的前提下，使充电费用最小化。针对该模型，提出了一种基于安全深度强化学习(SDRL)的无模型求解方法。在[31]中，作者的目标是协调电动汽车调度以缓解有功功率波动。
​

CS推荐是决定用户体验质量(QoE)的重要指标，近年来受到了广泛的关注。本文从推荐系统/方案设计和推荐建模两个方面对前人的工作进行了综述。
​

文献[32]提出了一个由EVs、CSs和全局聚合器(GA)组成的推荐系统，其中GA是一个集中的实体，它利用CSs的条件信息和EVs的计费预约进行CS推荐。本文在文[8]中进一步扩展了这一工作，提出了一种由电动汽车、移动边缘计算(MEC)和云计算(CC)组成的电动汽车充电分层体系结构。文[5]研究了电动汽车出租车CS推荐系统。该系统包括三个核心模块：状态推理、等待时间计算和CS推荐。该系统根据电动出租车的充值意图，计算每个CS的等待时间，并为出租车选择最佳的等待时间。
​

在[9]中，作者定义了一个两级推荐方案，其中较低的一级是充电率控制，它处理如何将CS的电源分配给电动汽车，而较高的一级是CS推荐，它分配了一个最佳站点。[33]中提出了一种充电管理方案，可为异构电动汽车提供抢占式充电服务。
​

推荐模型在确定总体充电性能方面起着至关重要的作用，这也是本文研究的重点。以前的工作将CS推荐问题描述为博弈论、LP问题、广义指派问题等。
​

在文献[6]中，推荐问题被建模为一个博弈论，其中每个EV都是一个参与者，在本文中使用了时间触发策略。即在每个预定义的时间瞬间，所有有充电需求的电动汽车选择一个能够最小化循环行驶和排队时间的CS。在此过程中，如果一个EV（如EV1)选择了一个使前一个策略（如EV2)变得非最优的策略，那么EV2必须重新选择它的策略。利用归纳法证明了该博弈存在纳什均衡。文献[13]的作者将推荐问题建模为一个Stackelberg博弈，考虑了收费价格、排队时间和CSS之间的距离。然而，博弈模型并不能保证获得推荐的最优解。
​

在文献[14]中，作者将CS推荐问题看作一个在线动态资源分配问题，并将其转化为一个具有公平性约束的Pareto优化问题。然后采用基于贪心的算法对模型进行求解。
​

LP（或IP）也用于CS推荐。Ni等人[15]研究了CS关于电池交换的建议。为了最大限度地提高所有CSs的总收入，推荐系统为用户推荐CSs列表，而不仅仅是一个CS，然后让用户选择一个。这样，可以减少推荐失败。然后将该推荐问题表述为LP问题，其中一些充满电的电池策略性地保留给能够接受更高价格的未来客户。类似地，Tan等人[16]也在研究电池交换-ping CS，但他们将其建模为一个混合整数线性规划（MILP），并开发了一个广义benders decom-position算法来求解该模型。
​

如上所述，CS推荐问题可以看作是一个指派问题。孔等人。[9]将其建模为一个广义指派问题。然后，他们提出了两个近似算法。一种算法是具有多项式复杂度的3-逼近，另一种算法是使用完全多项式时间逼近格式的(2+)逼近。文献[34]将车辆-CS分配问题建模为MILP。作者提出了一种代理辅助优化方法来求解所建立的模型。
​

由于推荐问题和调度问题紧密联系在一起，以往的工作也同时解决了这两个问题。在[35]中，提出的系统对每个收费请求进行重新计算和调度，以使总收入最大化。将该问题转化为一个优化问题，并采用一种称为投标-价格控制的启发式方法求解该模型。在[36]中，作者的目标是同时优化调度（充电时间）和推荐。他们通过最小化充电成本来安排电动汽车的充电时间，并通过最小化充电时间来推荐CS。[37]的作者使用博弈论方法设计了多组公共电动汽车竞争容量有限的充电站的时空调度。
​

由于推荐问题和调度问题紧密联系在一起，以往的工作也同时解决了这两个问题。在[35]中，提出的系统对每个收费请求进行重新计算和调度，以使总收入最大化。将该问题转化为一个优化问题，并采用一种称为投标-价格控制的启发式方法求解该模型。在[36]中，作者的目标是同时优化调度（充电时间）和推荐。他们通过最小化充电成本来安排电动汽车的充电时间，并通过最小化充电时间来推荐CS。[37]的作者使用博弈论方法设计了多组公共电动汽车竞争容量有限的充电站的时空调度。
# 3. 问题描述
我们考虑的EV充电基础设施的控制和管理的推荐系统。推荐系统的目的是通过对每个计费请求进行最佳推荐，使总计费时间最小化，并实现CSs之间的负载平衡。我们将电动汽车充电过程描述如下（图1）。
![image.png](https://cdn.nlark.com/yuque/0/2021/png/20357720/1636379943801-8ef4621c-dc75-430c-9d14-52ad58268010.png#clientId=u83e02f10-3884-4&from=paste&height=732&id=udec957c2&margin=%5Bobject%20Object%5D&name=image.png&originHeight=1464&originWidth=1892&originalType=binary&ratio=1&size=207225&status=done&style=none&taskId=u86cfa99f-88bf-442b-8d23-71a613b4b5f&width=946)

- 当EV耗尽其能量时，驾驶员通过例如APP向推荐系统发送充电请求；
- 在接收到该请求时，推荐系统根据系统的状态做出决定（选择CS）（状态定义请参阅第三节）；
- 推荐系统向请求EV回复推荐消息并更新状态；
- 当电动汽车到达推荐的CS时，CS通知推荐系统，由推荐系统更新状态；
- 当电动汽车完成充电时，CS再次通知推荐系统更新状态。

用Voronoi图划分整个充电区Z（图2）。对于CS集![](https://cdn.nlark.com/yuque/__latex/9c4747f2c83bb8047d769df612a4a189.svg#card=math&code=%5C%7Bc_1%2Cc_2%2C...%2Cc_%7Bn-1%7D%2Cc_n%5C%7D&id=Mjw7F)，其中![](https://cdn.nlark.com/yuque/__latex/7b8b965ad4bca0e41ab51de7b31363a1.svg#card=math&code=n&id=T7Cbr)是区域中CS的数目。![](https://cdn.nlark.com/yuque/__latex/96fafac0c054b9eb47d3f630ed02c289.svg#card=math&code=c_i&id=D1NV4)站的区域![](https://cdn.nlark.com/yuque/__latex/6cecb22625d972562491fcf2ef35ed39.svg#card=math&code=Ri&id=sN48g)定义为：
![](https://cdn.nlark.com/yuque/__latex/bb38c2f32f0774ce201cd806aaf4342c.svg#card=math&code=R_i%20%3D%5C%7Bl%E2%88%88Z%7Cd%28l%2Cc_i%29%3Cd%28l%2Cc_j%29%2Cj%E2%89%A4n%2Cj%04%5Cneq%20i%5C%7D&id=lWjkB)
![image.png](https://cdn.nlark.com/yuque/0/2021/png/20357720/1636380337474-aea40fcb-0897-4d56-8c7b-c411f93cb418.png#clientId=u83e02f10-3884-4&from=paste&height=555&id=ub0c73a7e&margin=%5Bobject%20Object%5D&name=image.png&originHeight=1110&originWidth=1668&originalType=binary&ratio=1&size=336043&status=done&style=none&taskId=uc9ef7e3b-c0db-47d2-9a0d-bd7d8d2a104&width=834)
与以往的推荐通常在固定的时间间隔（时间触发器）触发不同，我们的推荐是由计费请求（事件触发器）触发的。显然，事件触发机制比时间触发机制有优势，因为它可以立即接收推荐，而 后者应该等待触发时间。在我们的提议中，时间范围由EV的充电请求来划分，如图3所示：
![image.png](https://cdn.nlark.com/yuque/0/2021/png/20357720/1636638938947-65e164ed-1652-4129-b26a-049e045200c3.png#clientId=u638d0723-81d1-4&from=paste&height=341&id=u6da1ef33&margin=%5Bobject%20Object%5D&name=image.png&originHeight=682&originWidth=1854&originalType=binary&ratio=1&size=289543&status=done&style=none&taskId=u8771099f-7f3c-4343-92f1-5a1dfb1d35e&width=927)


充电请求到达的随机性使得决策周期（两个连续到达之间的间隔）发生变化。在每个决策时刻t，系统接收充电请求，并对CS建议做出决策，即请求电动汽车应驱动至哪个CS。在收到此建议后，车辆将花费![](https://cdn.nlark.com/yuque/__latex/388d9003a8c6b632355ca7d082e02f78.svg#card=math&code=%CF%84_d&id=KK6uq)的行驶时间与![](https://cdn.nlark.com/yuque/__latex/c0ecc73ad0a20fe712ff9dffa68059d1.svg#card=math&code=%CF%84_w&id=nCZPU)的排队时间（如果有可用的充电桩，![](https://cdn.nlark.com/yuque/__latex/288e3597fcc7b786fb63413957b50e40.svg#card=math&code=%CF%84_w%20%3D%200&id=xyf6d)）和![](https://cdn.nlark.com/yuque/__latex/50ea1c73e7aa6b1e0959f1558aaeaec9.svg#card=math&code=%CF%84_c&id=CCtta)的充电时间。
​

CS推荐（决策）是根据系统的状态信息做出的。它包括**CS容量**、**排队长度信息**、**请求电动汽车的信息**、**先前推荐但仍在路上的电动汽车的信息（到车站电动汽车）**。我们将系统状态定义为![](https://cdn.nlark.com/yuque/__latex/ccae86bcc7de0b3deb9ec8f947f1e0fb.svg#card=math&code=S%20%5Ctriangleq%20%28M%EF%BC%8CX%EF%BC%8CV%29&id=GiIw8)，其中![](https://cdn.nlark.com/yuque/__latex/69691c7bdcc3ce6d5d8a1361f22d04ac.svg#card=math&code=M&id=Kc9d8)表示CS的状态，![](https://cdn.nlark.com/yuque/__latex/02129bb861061d1a052c592e2dc6b383.svg#card=math&code=X&id=iZDOd)表示到站EVs的状态，![](https://cdn.nlark.com/yuque/__latex/5206560a306a2e085a437fd258eb57ce.svg#card=math&code=V&id=DWvh1)表示请求EV的状态。

- ![](https://cdn.nlark.com/yuque/__latex/67f06aa8c3043064b474094b6ee6714d.svg#card=math&code=M%3D%28m_1%2Cm_2%2C...%2Cm_n%29&id=DtgeN)是一个n元组，其元素![](https://cdn.nlark.com/yuque/__latex/342e772474b691ac87dac30aeef596c0.svg#card=math&code=m_i&id=nsFqw)为整数，![](https://cdn.nlark.com/yuque/__latex/d7f887c98e338949726d02b61983de3c.svg#card=math&code=m_i%E2%88%88%5B-k_i%2C%E2%88%9E%29&id=jglZP)，![](https://cdn.nlark.com/yuque/__latex/23067b73ec7d14340d9f2d34ded0e520.svg#card=math&code=i%E2%88%88%5B1%2Cn%5D&id=rgsCq)。其中![](https://cdn.nlark.com/yuque/__latex/34c1d173d638ceb8fb5bec184c055549.svg#card=math&code=k_i&id=k5KDm)是CS处的充电桩数。![](https://cdn.nlark.com/yuque/__latex/40f08811c46d895086aad80f9d59edf8.svg#card=math&code=M%3D0&id=MfEnZ)表示没有等待充电的电动汽车，但所有的充电桩都被占用了；![](https://cdn.nlark.com/yuque/__latex/72929aee87c7a21face4aaf6c9be13d7.svg#card=math&code=M%3C0&id=B0B67)表示有![](https://cdn.nlark.com/yuque/__latex/69691c7bdcc3ce6d5d8a1361f22d04ac.svg#card=math&code=M&id=mKAKA)个充电桩可用；![](https://cdn.nlark.com/yuque/__latex/eb124f0aa79b80ec79e5a87233e7ac37.svg#card=math&code=M%3E0&id=k8thj)表示（队列中）等待充电的电动汽车数。




- ![](https://cdn.nlark.com/yuque/__latex/fc05ef3c9d7edba083778068744c3b99.svg#card=math&code=X%3D%28x_1%2Cx_2%2C...%2Cx_n%29&id=bVHTs)也是一个n元组，它的元素![](https://cdn.nlark.com/yuque/__latex/01f57c20d969274be0b98c2d6bfb7fae.svg#card=math&code=x_i%E2%88%88%5B0%2C%E2%88%9E%29&id=R1ddv)表示已经分配给CS ![](https://cdn.nlark.com/yuque/__latex/865c0c0b4ab0e063e5caa3387c1a8741.svg#card=math&code=i&id=ltmft) 但仍在路上的EVs的个数。




- ![](https://cdn.nlark.com/yuque/__latex/86770a947e240871e01f0c1e5b9c275b.svg#card=math&code=V%3D%28l%2Ce%29&id=lfP77)是一个二元组，其中![](https://cdn.nlark.com/yuque/__latex/2db95e8e1a9267b7a1188556b2013b33.svg#card=math&code=l&id=n4WFu)表示电动汽车请求充电时的位置；![](https://cdn.nlark.com/yuque/__latex/e1671797c52e15f763380b45e841ec32.svg#card=math&code=e&id=xOsNI)代表需要充电的电量。我们不使用坐标，而是设置代表归属区域![](https://cdn.nlark.com/yuque/__latex/c42952440ffcf8129a1bedeac4773415.svg#card=math&code=R_i&id=VfADf)的位置信息![](https://cdn.nlark.com/yuque/__latex/77cea048564e1fec4db5b83f62a45899.svg#card=math&code=l%E2%88%88%5B1%2Cn%5D&id=Nh40I)。



在时刻![](https://cdn.nlark.com/yuque/__latex/e358efa489f58062f10dd7316b65649e.svg#card=math&code=t&id=no9x1)收到充电请求后，系统应根据其当前状态![](https://cdn.nlark.com/yuque/__latex/86ad9159785a8f6f1c1a74c4eac26365.svg#card=math&code=s_t&id=jIy0g)，![](https://cdn.nlark.com/yuque/__latex/4e9ffddccda8e0802dceb8b71ea8ac37.svg#card=math&code=s_t%20%5Cin%20S&id=Wj4jC)作出决定。 即采取应向请求车辆推荐CS的行动。因此，我们定义了动作集![](https://cdn.nlark.com/yuque/__latex/3e169e2b61ade4fcba70edcd971614f1.svg#card=math&code=A%20%5Ctriangleq%20%5B1%2Cn%5D&id=aKawf)和动作![](https://cdn.nlark.com/yuque/__latex/c8830b2d4a302408ef3613e5bf68cc76.svg#card=math&code=a%E2%88%88%20A&id=gOZ73)。![](https://cdn.nlark.com/yuque/__latex/2c09b56d166e2f375c8dc6f38a27b7e0.svg#card=math&code=a%3Di&id=yAdB5)表示向请求EV推荐CS ![](https://cdn.nlark.com/yuque/__latex/865c0c0b4ab0e063e5caa3387c1a8741.svg#card=math&code=i&id=I9eEv)。在采取动作![](https://cdn.nlark.com/yuque/__latex/2c09b56d166e2f375c8dc6f38a27b7e0.svg#card=math&code=a%3Di&id=WbITo)之后，系统以概率![](https://cdn.nlark.com/yuque/__latex/229363194bc787209173d24205a40a69.svg#card=math&code=p%28s_%7Bt%2B1%7D%7Cs_t%2Ca%29&id=qkXtD)移动到下一个状态![](https://cdn.nlark.com/yuque/__latex/b4527e0091094c13aa10ddd6a0ab36be.svg#card=math&code=s_%7Bt%2B1%7D&id=T86rT)。系统将获得奖励![](https://cdn.nlark.com/yuque/__latex/9cfd718d20a671debf4cf29f15adcd86.svg#card=math&code=r_a%28s_t%29&id=JVt9s)，其定义为总充电时间的倒数，总充电时间是从车辆请求充电到完成充电的时间间隔。
![](https://cdn.nlark.com/yuque/__latex/5852553a631dc9199930623a186a2b72.svg#card=math&code=r_a%28s_t%29%3D%5Cfrac%7B1%7D%7B%5Ctau_d%20%2B%20%5Ctau_%7B%5Comega%7D%20%2B%20%5Ctau_c%7D&id=C3kbv)
MDP中的策略是一个映射![](https://cdn.nlark.com/yuque/__latex/a47aee14e69dee10762772758548fc45.svg#card=math&code=%CF%80%3AS%E2%86%92A&id=yu9Mm)。如果使用策略![](https://cdn.nlark.com/yuque/__latex/31bf0b12546409e15021243132fc7574.svg#card=math&code=%CF%80&id=N6yhl)，并且状态![](https://cdn.nlark.com/yuque/__latex/5dbc98dcc983a70728bd082d1a47546e.svg#card=math&code=S&id=FojXO)是初始状态，则状态值函数![](https://cdn.nlark.com/yuque/__latex/0f5738e501341b9cf27dcdff58b2b6b6.svg#card=math&code=V%5E%CF%80%28s%29&id=jYIlp)是期望的总收益。
![](https://cdn.nlark.com/yuque/__latex/dc82fed702055f78eacab750eeefffcd.svg#card=math&code=v%5E%5Cpi%28s%29%3DE%5E%5Cpi_s%5C%7B%5Csum%5E%7B%5Cinfty%7D_%7Bt%3D1%7D%20%5Clambda%5E%7Bt-1%7Dr_a%28s_t%29%20%5C%7D%2Cs_t%20%5Cin%20S%2C%20a_t%20%5Cin%20%5Cmathcal%7BA%7D%20%5Cqquad%20%281%29&id=Acc1U)
其中![](https://cdn.nlark.com/yuque/__latex/d242838df85274ef228794bb684810a6.svg#card=math&code=s_1%3Ds&id=B8uge)，![](https://cdn.nlark.com/yuque/__latex/b2a3752a8ba12ce39ddd915714f14420.svg#card=math&code=%CE%BB%28%3C1%29&id=fBoLo)是一个不变的折扣因子。设![](https://cdn.nlark.com/yuque/__latex/0a07ea387f06fdb2fae6eda52896dae0.svg#card=math&code=v%5E%2A%28s%29&id=uf9vE)表示在最优策略下对状态s的奖励，理论上可以通过递归求解下面的Bellman方程得到它：
![](https://cdn.nlark.com/yuque/__latex/64fdb0f0e27b04f8cdb5b4b844ec22a2.svg#card=math&code=v%5E%2A%28s%29%3D%5Cmax%5Climits_%7Ba%20%5Cin%20%5Cmathbb%7BA%7D%7D%5C%7B%0Ar_a%28s%29%2B%5Clambda%20%5Csum_%7Bs%27%7Dp%28s%27%7Cs%2Ca%29v%5E%2A%28s%27%29%0A%5C%7D%20%5Cqquad%20%282%29&id=jAtY8)
其中![](https://cdn.nlark.com/yuque/__latex/085c9f0df11642cf704f40ffdf753055.svg#card=math&code=s%27&id=w0zvX)是![](https://cdn.nlark.com/yuque/__latex/03c7c0ace395d80182db07ae2c30f034.svg#card=math&code=s&id=Rgxx4)的下一个状态。
​

然而，在我们的场景中求解上述Bellman（布尔曼）方程具有挑战性。挑战来自三个方面。首先确定![](https://cdn.nlark.com/yuque/__latex/6b7a9143b32a95234a8528c2c8d843f7.svg#card=math&code=r_a%28s%29&id=F4Bll)，它依赖于行车时间、排队时间和充电时间。即使我们可以假设驾驶时间和充电时间遵循一个I.I.D（独立同分布），计算电动汽车充电完成时间的期望是非常复杂的，因为电动汽车的排队时间受到其他电动汽车的影响，这些电动汽车也是以前推荐给本CS的，但可能在任何时候到达，即在本电动汽车之前或之后。
​

第二个是确定下一个状态。充电请求的到达和电动汽车的到达/离开站点的随机过程使得下一个状态（如![](https://cdn.nlark.com/yuque/__latex/6eb0cfbf5bd9742bfb8dfa966306ad3c.svg#card=math&code=s_t%2B1&id=Cq7LC)）非常不确定。例如，在t+1时刻，先前的车辆（需要在t、t-1,...充电）可能仍然在路上，或者已经在排队，或者正在充电，或者已经完成充电。因此，下一个状态的确定具有挑战性。
​

三是系统状态空间大。系统状态由（2n+2）个元素组成。如果n较大，并且充电能量元e（连续值）以细粒度方式离散，则状态空间非常大。
​

为了解决这些问题，我们提出的关键思想是将决策周期划分为多个阶段，以简化状态转换和状态空间复杂度。为此，我们使用PDS并在MDP中引入一个名为IDS的新状态来形成MMDP。在决策周期中，即从一个决策时刻到下一个决策时刻，系统从正常状态过渡到PDS，然后过渡到IDS，最后过渡到下一个正常状态。针对MMDP模型的求解问题，提出了一种在线学习算法。
​

# 4. 系统模型
一个决策周期内的三个阶段在我们的提议中被定义，如图4所示：
![image.png](https://cdn.nlark.com/yuque/0/2021/png/20357720/1636804951029-8a18beac-4549-424c-9819-7f386415fb67.png#clientId=u890968dc-0e4a-4&from=paste&height=299&id=uf2f99dd0&margin=%5Bobject%20Object%5D&name=image.png&originHeight=598&originWidth=1676&originalType=binary&ratio=1&size=256896&status=done&style=none&taskId=u7ce63b80-622b-4110-b32a-66467d154a8&width=838)
第一阶段介于正常状态s和PDS之间（我们用![](https://cdn.nlark.com/yuque/__latex/44b0f04acbbde994ea20795727b0d125.svg#card=math&code=%5Chat%7Bs%7D&id=rTqjH)表示PDS）；第二阶段在PDS和IDS之间（我们使用![](https://cdn.nlark.com/yuque/__latex/9731d0091ba3d996fd0899c50ff5bcc0.svg#card=math&code=%5Ctilde%7Bs%7D&id=wvEx3)表示IDS）；第三阶段是IDS和下一个正常状态之间的阶段。为了降低状态空间的复杂性，我们让不同的状态取![](https://cdn.nlark.com/yuque/__latex/68ee5331285b8ed88f275b07a4103477.svg#card=math&code=%28M%EF%BC%8CX%EF%BC%8CV%29&id=bO3nY)的子集。正常状态重新定义为![](https://cdn.nlark.com/yuque/__latex/91dacd64d77c0ec7c8907863623f9500.svg#card=math&code=%5Cmathbb%7BS%7D%20%5Ctriangleq%20%28M%2CV%29&id=HCLrV)。这里，我们省略了到站电动汽车的状态，但让它在PDS和IDS中考虑。PDS和IDS定义相同，即![](https://cdn.nlark.com/yuque/__latex/6d1d45af6ce76518bd985ca14e784dc6.svg#card=math&code=%5Chat%7BS%7D%20%5Ctriangleq%28M%EF%BC%8CX%29&id=MPNJT)，![](https://cdn.nlark.com/yuque/__latex/e2435c8ad98fc269462d3af1cbf9b5e9.svg#card=math&code=%5Ctilde%7BS%7D%20%5Ctriangleq%28M%EF%BC%8CX%29&id=RLjeB)。
​

在我们的模型中，PDS是决策后的即时状态。例如，在三个CSs(c1，c2，c3)的场景中，我们假设当前到站的电动汽车信息为![](https://cdn.nlark.com/yuque/__latex/206dc31b3e3903bae6abaf287ec21d76.svg#card=math&code=x_t%3D%283%2C2%2C0%29&id=dldPL)，表示分别有3，2和0辆电动汽车在前往c1、c2和c3的途中。决策时刻的正常状态为![](https://cdn.nlark.com/yuque/__latex/8551952abb475f34c03237851cd65d74.svg#card=math&code=s_t%3D%28%282%2C3%2C3%29%2C%282%2C5%29%29&id=RzKf0)。根据该状态，系统向当前充电请求推荐c3。根据该判决，PDS被设置为![](https://cdn.nlark.com/yuque/__latex/a994f9b6739f5f843d45a5afe0c3859b.svg#card=math&code=%5Chat%7Bs%7D%3D%28%282%2C3%2C3%29%2C%283%2C2%2C1%29%29&id=OQFOR)。根据这个定义，我们省略了关于请求EV ![](https://cdn.nlark.com/yuque/__latex/d8584311108d7851f1594414a743063e.svg#card=math&code=%5Cmathcal%7BV%7D&id=FoV7L)的状态信息，因为这些信息在做出决定后是无用的。通过引入PDS，可以在第一阶段限制作用对状态的影响。Bellman方程（2）改写为：
![](https://cdn.nlark.com/yuque/__latex/45a73d834462982bfbed09f839769083.svg#card=math&code=v%5E%2A%28s%29%3D%5Cmax%5Climits_%7Ba%20%5Cin%20%5Cmathcal%7BA%7D%7D%5C%7B%0Ar_a%28s%29%2B%5Clambda%20v%5E%2A%28%5Chat%7Bs%7D%29%0A%5C%7D%20%5Cqquad%20%283%29&id=lDvcM)
在上面的方程中，由于定义的PDS是一个动作的确定性状态，所以概率被移除了。
​

与EV请求触发的正常状态不同，IDS是由EV到达或离开CS触发的。在上面的两个CS场景中，假设一辆电动汽车（前面已经推荐过）到达c1。此时的状态是![](https://cdn.nlark.com/yuque/__latex/18ef8a92c6ca4bd4713b2918ef010fca.svg#card=math&code=%5Chat%7Bs%7D%5E1_t%20%3D%20%28%283%2C3%2C3%29%2C%282%2C2%2C1%29%29&id=e0upJ)。然后，在c2处的电动汽车完成充电并离开，因此![](https://cdn.nlark.com/yuque/__latex/d5ff5f735adf1fcad5e6bf5e4b2bb36f.svg#card=math&code=%5Ctilde%7Bs%7D%5E2_t%3D%28%283%2C2%2C3%29%2C%282%2C2%2C1%29%29&id=lFrZ2)。然后，一个新的充电请求（3,4）到达，系统进入正常状态![](https://cdn.nlark.com/yuque/__latex/1b4c62144180345bc60138523122e94d.svg#card=math&code=s_%7Bt%2B1%7D%20%3D%20%28%283%2C2%2C3%29%2C%283%2C4%29%29&id=t7Lk3)。
​

在我们的MMDP模型中，在每个决策阶段，我们只考虑下一个决策时刻之前的最后一个中间状态。因此，在上面的例子中，模型中只考虑了![](https://cdn.nlark.com/yuque/__latex/432e3a717b050dfa155e5e20a1dc8278.svg#card=math&code=%5Ctilde%7Bs%7D%5E2_t&id=z2y0X)并将其视为IDS。从PDS ![](https://cdn.nlark.com/yuque/__latex/44b0f04acbbde994ea20795727b0d125.svg#card=math&code=%5Chat%7Bs%7D&id=M7oJs)到IDS![](https://cdn.nlark.com/yuque/__latex/9731d0091ba3d996fd0899c50ff5bcc0.svg#card=math&code=%5Ctilde%7Bs%7D&id=agnOK)的转换与动作无关。实际上，它取决于CSs和到站电动汽车的状态信息。这就是为什么两种状态都被定义为这些信息的组合。然后，我们定义PDS的价值函数如下：
![](https://cdn.nlark.com/yuque/__latex/ba41532945bffb66b104e35f54eaa90f.svg#card=math&code=v%5E%2A%28%5Chat%7Bs%7D%29%3D%5Csum_%7B%5Ctilde%7Bs%7D%7Dp%28%5Ctilde%7Bs%7D%7C%5Chat%7Bs%7D%29v%5E%2A%28%5Ctilde%7Bs%7D%29%5Cforall%20%5Ctilde%7Bs%7D%20%20%5Cqquad%20%284%29&id=I7lUL)
其中，![](https://cdn.nlark.com/yuque/__latex/23f078e9ed37850ca5fa1d280bf3ef84.svg#card=math&code=p%28%5Ctilde%7Bs%7D%7C%5Chat%7Bs%7D%29&id=uHQZa)是从状态PDS到状态IDS的转移概率，它只取决于电动汽车到达和离开一个车站。因此，在第二阶段，电动汽车的到达和离开对状态的影响是有限的。
​

最后，通过充电请求触发从IDS ![](https://cdn.nlark.com/yuque/__latex/03f51fdab9d944f698a912d2805eb644.svg#card=math&code=%5Ctilde%7Bs%7D_t&id=HZilb)到下一个正常状态![](https://cdn.nlark.com/yuque/__latex/b4527e0091094c13aa10ddd6a0ab36be.svg#card=math&code=s_%7Bt%2B1%7D&id=LBbvH)的转换。此转换保持CS的状态信息不变，即![](https://cdn.nlark.com/yuque/__latex/00bbfa293e32b8da70bcf09dd3d5fe9a.svg#card=math&code=s.m_i%3D%5Ctilde%7Bs%7D.m_i%20%5Cforall%20i&id=E4Pmp)。然后，PDS的价值函数定义为：
![](https://cdn.nlark.com/yuque/__latex/42c2b81320c19fa71f0790cb6db0ed07.svg#card=math&code=v%5E%2A%28%5Ctilde%7Bs%7D%29%20%3D%20%5Csum_%7Bs%7Dp%28s%7C%5Ctilde%7Bs%7D%29v%5E%2A%28s%29%20%5C%5C%20s%20%3D%20%5C%7B%20x%20%5Cin%20%5Cmathbb%7BS%7D%7Cx.m_1%3D%5Ctilde%7Bs%7D.m_1%2C...%2Cx.m_n%3D%5Ctilde%7Bs%7D.m_n%20%5C%7D%20%5Cqquad%20%285%29%20&id=xC8ar)
其中![](https://cdn.nlark.com/yuque/__latex/a317ef835281fddb4c3b0faf0a10f24f.svg#card=math&code=p%28s%7C%5Ctilde%7Bs%7D%29&id=Hg6HO)是从状态IDS到正常状态的转移概率，它只依赖于充电请求的到达。因此，在第三阶段，充电请求的到达对状态的影响是有限的。
​

在CS推荐系统中，状态信息的变化可以由事件（例如EV的到达）触发，并且是可观察的，这使得可以将MDP状态的转变分成多个阶段。MMDP模型的一个好处是降低了复杂性，因为所有类型的已定义状态![](https://cdn.nlark.com/yuque/__latex/ea4d3000ff53e3d10d336ffcfaf253f5.svg#card=math&code=%28%5Cmathbb%7BS%7D%2C%5Chat%7B%5Cmathbb%7BS%7D%7D%2C%5Ctilde%7B%5Cmathbb%7BS%7D%7D%29&id=T2k0G)都是原始状态![](https://cdn.nlark.com/yuque/__latex/575eea3a71bb9a4b911654aa4b00df42.svg#card=math&code=%28%5Cmathcal%7BS%7D%29&id=Ryvcd)的子集。将简化的正常状态![](https://cdn.nlark.com/yuque/__latex/8e60bf923d4f9354c5504432c56e2cb3.svg#card=math&code=%28%28M%2CX%2CV%29%29&id=UjZgG)简化为![](https://cdn.nlark.com/yuque/__latex/c137e7debd63ac7ad5fc16df42cde894.svg#card=math&code=%28M%2CV%29%29&id=smuKS)，可能会降低推荐的准确性。然而，公式化的Bellman方程（3）的右侧包括所有状态信息，即CSs的状态信息、请求EV的状态信息和到达站EV的状态信息。有了所有这些必要的信息，推荐系统就有可能做出最优推荐。通过将决策周期划分为三个阶段，将原Bellman方程（2）分解为三个方程(3)、(4)和（5）。与Bellman方程（2）相比，在Bellman方程（2）中，执行最大化需要知道概率![](https://cdn.nlark.com/yuque/__latex/cb9c32628060a2f538bb4fa1d94d6609.svg#card=math&code=p%28s%27%7Cs%EF%BC%8Ca%29&id=C5Nel)，而这几乎是不可能获得的，我们的命题消除了最大化操作中的转移概率（Bellman方程(3))，并简化了转移概率（（4）和（5））。利用这个多阶段模型，我们可以设计一个高效的在线算法来获得最优解。
​

# 5. 在线算法
在我们的建议中，系统应该了解实时状态信息。当事件触发时，CSs和EV会对此进行更新，例如，EV到达/离开CS或充电请求到达。例如，当EV到达CS时，CS向系统发送此事件。后者更新到站点EV的状态信息和CS的排队信息。


通过将状态转移分解为三个阶段，我们提出的MMDP模型可以通过基于在线学习的算法简单而精确地求解。反过来，在线算法不需要先验的转移概率知识，很好地符合CS推荐的MDP模型。
## 5.1 基于在线学习的算法
基于在线学习的算法使系统能够迭代更新每个状态的值函数，并最终收敛到最优解。首先，我们将（3）改写为迭代形式：
![](https://cdn.nlark.com/yuque/__latex/dabfb7fc05c6491f38083e78d0284f62.svg#card=math&code=v_%7Bt%2B1%7D%28s%29%3D%5Cmax%5Climits_%7Ba%20%5Cin%20%5Cmathbb%7BA%7D%7D%5C%7B%0Ar_%7Bt%2B1%7D%28s%2Ca%29%2B%5Clambda%20v_t%28%5Chat%7Bs%7D%29%0A%5C%7D%20%5Cqquad%20%286%29&id=s3QSr)
然后，我们需要从（4）和（5）中删除转移概率，这是一个不重要的工作，因为概率已经移到最大运算之外。
​

设![](https://cdn.nlark.com/yuque/__latex/27d14f1ab305e3bb07a52d58a292818b.svg#card=math&code=%5Crho%20_t&id=V8qJG)为具有以下属性的正更新序列：
![](https://cdn.nlark.com/yuque/__latex/1355edb8bd0322f222f03e43ef42350a.svg#card=math&code=%5Csum_%7Bt%3D0%7D%5E%7B%5Cinfty%7D%20%5Crho_t%20%3D%20%5Cinfty%3B%20%5Cquad%20%0A%5Csum_%7Bt%3D0%7D%5E%7B%5Cinfty%7D%28%5Crho_t%29%5E2%20%3C%20%5Cinfty%0A%5Cqquad%20%287%29&id=p63AW)
然后，根据随机逼近理论[38]，PDS ![](https://cdn.nlark.com/yuque/__latex/44b0f04acbbde994ea20795727b0d125.svg#card=math&code=%5Chat%7Bs%7D&id=XL5Zb)（4）的价值函数可更新为：
![](https://cdn.nlark.com/yuque/__latex/e80380d7b17d2c6f9801e15551ad313a.svg#card=math&code=v_%7Bt%2B1%7D%28%5Chat%7Bs%7D%29%3D%281-%5Crho_t%29v_t%28%5Chat%7Bs%7D%29%2B%5Crho_tv_%7Bt%2B1%7D%28%5Ctilde%7Bs%7D%29%20%5Cforall%20%5Ctilde%7Bs%7D%20%5Cqquad%20%288%29&id=qey29)
IDS ![](https://cdn.nlark.com/yuque/__latex/36f67e00e94829d9b23950fba386794d.svg#card=math&code=%5Ctilde%7Bs%7D%EF%BC%885%EF%BC%89&id=lxA1B)的价值函数可以更新为：
![](https://cdn.nlark.com/yuque/__latex/df57942c097c11e0207e17d77e7f2c0a.svg#card=math&code=v_%7Bt%2B1%7D%28%5Ctilde%7Bs%7D%29%3D%281-%5Crho_t%29v_t%28%5Ctilde%7Bs%7D%29%2B%5Crho_tv_%7Bt%2B1%7D%28s%29%20%5Cforall%20s%20%5Cqquad%20%289%29&id=F2NG3)
算法1描述了CS推荐的在线算法。在每个决策时刻![](https://cdn.nlark.com/yuque/__latex/865c0c0b4ab0e063e5caa3387c1a8741.svg#card=math&code=i&id=fOaAx)，系统观察当前状态![](https://cdn.nlark.com/yuque/__latex/e406ac4d7c470823a8619c13dd7101be.svg#card=math&code=s_i&id=zzqKe)（步骤4），并为此状态选择最佳动作（步骤5）。基于所选择的动作，系统可以确定PDS ![](https://cdn.nlark.com/yuque/__latex/676d360a1d27d4d3113011d0179aa2ab.svg#card=math&code=%5Chat%7Bs_i%7D&id=F5LEm)（步骤6）。之后，系统继续观察系统状态，直到下一个决策时刻，并记录IDS ![](https://cdn.nlark.com/yuque/__latex/909930df362b43f2fccf75ceaeebd376.svg#card=math&code=%5Ctilde%7Bs_i%7D&id=j7RJN)（步骤7，提醒IDS是下一个决策时刻之前的最后一个中间状态）。在此决策期间，如果系统接收到任何充电终止信息，它将计算总充电时间并更新相关即时奖励![](https://cdn.nlark.com/yuque/__latex/6b7a9143b32a95234a8528c2c8d843f7.svg#card=math&code=r_a%28s%29&id=sEZxD)（步骤8）。使用在线算法，我们在第三节中讨论![](https://cdn.nlark.com/yuque/__latex/6b7a9143b32a95234a8528c2c8d843f7.svg#card=math&code=r_a%28s%29&id=SlXw0)的确定的挑战与简单的解决方法。
![image.png](https://cdn.nlark.com/yuque/0/2021/png/20357720/1636806009098-d1cfe28e-6ecc-458d-9aa5-328c3795a09d.png#clientId=u890968dc-0e4a-4&from=paste&height=643&id=u630edb68&margin=%5Bobject%20Object%5D&name=image.png&originHeight=1286&originWidth=1478&originalType=binary&ratio=1&size=221896&status=done&style=none&taskId=u61db4b76-e3ab-456f-b559-a9a4fad3c87&width=739)
​

最后，系统应该分别使用（3）、（9）和（8）更新![](https://cdn.nlark.com/yuque/__latex/03c7c0ace395d80182db07ae2c30f034.svg#card=math&code=s&id=jcq7p)，![](https://cdn.nlark.com/yuque/__latex/44b0f04acbbde994ea20795727b0d125.svg#card=math&code=%0A%5Chat%7Bs%7D&id=OyPlX)和![](https://cdn.nlark.com/yuque/__latex/9731d0091ba3d996fd0899c50ff5bcc0.svg#card=math&code=%5Ctilde%7Bs%7D&id=iiPHc)的值函数。注意，在迭代![](https://cdn.nlark.com/yuque/__latex/865c0c0b4ab0e063e5caa3387c1a8741.svg#card=math&code=i&id=QiMEq)中，我们得到![](https://cdn.nlark.com/yuque/__latex/d1ae364668146dec0496d05e55e58d04.svg#card=math&code=v%28s_i%29&id=XoF0C)（步骤9）。该值用于更新的![](https://cdn.nlark.com/yuque/__latex/e406ac4d7c470823a8619c13dd7101be.svg#card=math&code=s_i&id=eS7N3)前一个IDS ![](https://cdn.nlark.com/yuque/__latex/de466ac6997f514022224ed741c1f249.svg#card=math&code=%5Ctilde%7Bs%7D_%7Bi-1%7D&id=btyHc)（步骤10)，并且类似于PDS ![](https://cdn.nlark.com/yuque/__latex/de466ac6997f514022224ed741c1f249.svg#card=math&code=%5Ctilde%7Bs%7D_%7Bi-1%7D&id=yZ9JS)（步骤11）。在迭代![](https://cdn.nlark.com/yuque/__latex/865c0c0b4ab0e063e5caa3387c1a8741.svg#card=math&code=i&id=TZdnp)中确定的IDS和PDS的状态将在迭代![](https://cdn.nlark.com/yuque/__latex/15ab2d2b0b92c13f328635e5c4bdbe64.svg#card=math&code=i%2B1&id=c9rXk)中更新。
​

如前所述，用原有的MDP模型很难获得状态转移概率。特别是对于我们的事件触发MDP模型，转移概率可以是连续的。然而，通过分解状态转移，可以使用在线学习算法来评估状态值函数，而无需计算转移概率。
​

有两种方法来实施拟议的系统。一种是利用历史数据或模拟数据对系统进行训练，然后在现场部署。另一种是直接部署在线算法。但是，可以使用一些更有效的策略来代替一开始的随机推荐，例如，当系统还没有学习到任何东西时，可以向电动汽车推荐最近的CS（或最欠载的CS）。

