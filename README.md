
# Introduction

The *housing_adjustment* project implements my 2013 paper **The Effects of Housing Adjustment Costs on Consumption Dynamics**. That working paper is available on the Office of Financial Research's Staff Discussion Paper website ([link](https://www.financialresearch.gov/staff-discussion-papers/files/OFRsdp2015-03_Effects-of-Housing-Adjustment-Costs-on-Consumption-Dynamics.pdf)).

 The code is all in Matlab . The basic approach is [value function iteration](http://www.wouterdenhaan.com/numerical/VFIslides.pdf). Due to non-linearity in the optimal policy function and the desire for a precisely estimated solution, a value function iteration would be too slow. I make use of improvements to improve performance. First, it designed to run in parallel and scale well to a large number of nodes. Second, it makes use of [dynamic grid points](http://www.sciencedirect.com/science/article/pii/S0165176505003368) and [Howard's improvement algorithm](http://individual.utoronto.ca/zheli/policyfun.pdf). Third, I replaced Matlab code with C code in the most performance intensive part of the code. 
