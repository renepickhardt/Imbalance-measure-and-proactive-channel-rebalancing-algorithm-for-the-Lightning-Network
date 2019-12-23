The paper as a pdf can be found on arxiv: https://arxiv.org/abs/1912.09555 in the main folder of this repository you will also find the tex source files:

# Abstract
Making a payment in a privacy-aware payment channel network is achieved by trying several payment paths until one succeeds. With a large network, such as the Lightning Network, a completion of a single payment can take up to several minutes. We introduce a network imbalance measure and formulate the optimization problem of improving the balance of the network as a sequence of rebalancing operations of the funds within the channels along circular paths within the network. As the funds and balances of channels are not globally known, we introduce a greedy heuristic with which every node despite the uncertainty can improve its own local balance. In an empirical simulation on a recent snapshot of the Lightning Network we demonstrate that the imbalance distribution of the network has a Kolmogorov-Smirnoff distance of 0.74 in comparison to the imbalance distribution after the heuristic is applied. We further show that the success rate of a single unit payment increases from 11.2% on the imbalanced network to 98.3% in the balanced network. Similarly, the median possible payment size across all pairs of participants increases from 0 to 0.5 mBTC for initial routing attempts on the cheapest possible path. We provide an empirical evidence that routing fees should be dropped for proactive rebalancing operations. Executing 4 different strategies for selecting rebalancing cycles lead to similar results indicating that a collaborative approach within the friend of a friend network might be preferable from a practical point of view 

# Code

Note that I did some preliminary testing so the interesting code which was used to conduct the simulation, evaluation and experiments is in the `vs` subfolder of the code folder: https://github.com/renepickhardt/Imbalance-measure-and-proactive-channel-rebalancing-algorithm-for-the-Lightning-Network/tree/master/code/vs

As time was short the code is not in the best shape. I often changed the files from which and to which I saved the data of the experiments by hand and hardcoded when running several experiments. In this why you can't expect to just run the experiments with one command. Yet I think the code is valuable. 

# Data

The data was taken from the Gossip store of c-lighting. Best to install c-lightning and connect to a node to get your copy of the gossip store (for example you can find my lightning node at https://ln.rene-pickhardt.de )

The network was extracted from gossip and prepared with this script: https://github.com/renepickhardt/Imbalance-measure-and-proactive-channel-rebalancing-algorithm-for-the-Lightning-Network/blob/master/code/extractln.py (run `lightning-cli listchannels > channels.json` first to have the file)

# Results
The results can be found in `code/vs/fig` at https://github.com/renepickhardt/Imbalance-measure-and-proactive-channel-rebalancing-algorithm-for-the-Lightning-Network/tree/master/code/vs/fig 

# Consulting
Feel free to reach out if you need help to manage your lightning nodes or if you plan projects using the lightning network.

# Support
If you want to support my work please check out https://tallyco.in/s/lnbook or https://patreon.com/renepickhardt 
