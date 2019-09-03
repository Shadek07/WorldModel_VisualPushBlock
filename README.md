# WorldModel_VisualPushBlock
World Model implementation for Visual PushBlock Unity3D game with Discrete action space.

This repository is about applying worldmodel concept for visual pushblock Unity3d game. The first step is to create visual pushblock unity env with visual observation with 64x64 size. The environment will also contain some vector observations - positions of block, agent and goal; distances between agent and block, also between block and goal. One [blog post] (https://shadekcse.wordpress.com/2019/07/30/how-to-create-visualpushblock-unity-environment-executable-and-interact-with-that-in-python/) on env creation. Add the following lines at the end of CollectObservations function inside PushAgentBasics.cs file:
```
AddVectorObs(blockRB.position); //size 3
AddVectorObs(agentRB.position); //size 3
AddVectorObs(goal.transform.position); //size 3
AddVectorObs(Vector3.Distance(agentRB.position, blockRB.position)); //size 1
AddVectorObs(Vector3.Distance(goal.transform.position, blockRB.position)); //size 1
```
[Technical note] (https://shadekcse.wordpress.com/2019/08/02/how-to-replicate-world-model-carracing-by-training-from-scratch-on-ubuntu-18-04/) on replicating worldmodel can be used to replicate visual pushblock too.
