/********************
Author: Erik Talvitie
Date: 2017
********************/

#include "ConvolutionalBinaryCTS.h"
#include "ShooterModel.h"

#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include <boost/tuple/tuple.hpp>
#include <boost/unordered_map.hpp>
#include <boost/functional/hash.hpp>

using namespace std;
using namespace boost;

/* Takes the most recent observation and action and
   gives the resulting reward.*/
int rewardFunction(const vector<int>& obs, int act)
{
   int r = 0;
   if(act == 3)
   {
      r -= 1;
   }

   for(int target = 0; target < 3; target++)
   {
      int bottomLeft = 4*15 + target*5 + 1;
      if(obs[bottomLeft] && !obs[bottomLeft + 1] && obs[bottomLeft + 2] && 
	 !obs[bottomLeft - 15] && obs[bottomLeft - 14] && !obs[bottomLeft - 13])
      {
	 r += 10;
      }

      if(!obs[bottomLeft] && obs[bottomLeft + 1] && !obs[bottomLeft + 2] && 
	 obs[bottomLeft - 15] && !obs[bottomLeft - 14] && obs[bottomLeft - 13])
      {
	 r += 20;
      }
   }
   return r;
}

//Useful struct for MCTS planning
struct MCTSNode
{
   vector<double> summedReturns;
   vector<int> counts;
   int totalCount;
   vector<unordered_map<size_t, int> > children;
   int parent;
   MCTSNode(int numActions, int parentIndex){summedReturns.resize(numActions, 0); counts.resize(numActions, 0); children.resize(numActions); totalCount=0; parent=parentIndex;}
};

/* Takes an unrolled model, discount factor, current observation
   and uses Monte Carlo Tree Search to choose an action.
   Optional parameters:
   maxD - limits the depth of model to use. For rollout steps
   beond maxD, will simply repeat the maxDth model. Pass -1 to
   use the entire depth of the model.
   printRollouts - if true, prints things for debugging.*/
int mcts(const vector<ConvolutionalBinaryCTS*>& model, double discountFactor, const vector<int>& curObs, int maxD=-1, bool printRollouts=false, int firstAction=-1, int* rolloutActions=0, int* rolloutRewards=0, vector<int>* rolloutObs=0)
{
   int numRollouts = 200;
   int rolloutDepth = 15; //Hardcoding these for now. Probably ought to be parameters...
   int maxModelDepth = maxD;
   if(maxD < 0 || maxD > int(model.size()))
   {
      maxModelDepth = model.size();
   }

   int numActions = model[0]->getNumActs();

   vector<MCTSNode> tree(1, MCTSNode(numActions, -1));
   int numActionRollouts = 0;   
   for(int rollout = 0; rollout < numRollouts; rollout++)
   {
      for(int m = 0; m < maxModelDepth; m++)
      {
	 model[m]->saveState();
      }

      int t = 0;
      int m = 0;

      vector<int> obs = curObs;
      vector<int> rewards;
      vector<int> actions;
      int curNodeIndex = 0;
      double rolloutReturn = 0;      
      bool saveRollout = false;	       
      //Move down the tree until a node is missing a child
      while(t < rolloutDepth && tree[curNodeIndex].totalCount >= numActions)
      {      
	 vector<int> maxActs;
	 double maxScore = -numeric_limits<double>::infinity();
	 for(unsigned a = 0; a < numActions; a++)
	 {
	    double score = tree[curNodeIndex].summedReturns[a]/tree[curNodeIndex].counts[a] + 4*sqrt(log(tree[curNodeIndex].totalCount)/tree[curNodeIndex].counts[a]);
	    if(score-maxScore > 1e-6)
	    {
	       maxActs.clear();
	       maxActs.push_back(a);
	       maxScore = score;
	    }
	    else if(fabs(score-maxScore) <= 1e-6)
	    {
	       maxActs.push_back(a);
	    }	 
	 }
	 int action = maxActs[rand()%maxActs.size()];
	 int reward = rewardFunction(obs, action);
	 rewards.push_back(reward);
	 actions.push_back(action);

	 if(t == 0 && action == firstAction)
	 {
	    numActionRollouts++;
	    int r = rand()%numActionRollouts;
	    if(r == 0)
	    {
	       saveRollout = true;
	    }
	 }

	 if(saveRollout)
	 {
	    rolloutActions[t] = action;
	    rolloutRewards[t] = reward;
	    rolloutObs[t] = obs;
	 }
	 
	 bool endEpisode;
	 bool dummyReward;
	 model[m]->sample(action, obs, dummyReward, endEpisode);
	 if(m+1 < maxModelDepth)
	 {
	    m++;
	 }
	 if(t < rolloutDepth-1)
	 {
	    model[m]->update(action, obs, dummyReward, endEpisode, false);
	 }

	 size_t obsHash = hash_range(obs.begin(), obs.end());
	 int childIndex = tree[curNodeIndex].children[action][obsHash];
	 if(childIndex == 0) //first time seeing this observation, need a new node
	 {
	    tree.push_back(MCTSNode(numActions, curNodeIndex));
	    childIndex = tree.size()-1;
	    tree[curNodeIndex].children[action][obsHash] = childIndex;
	 }

	 curNodeIndex = childIndex;
	 if(printRollouts)
	 {
	    cout << "In tree A: " << action << endl;
	    for(int y = 0; y < 15; y++)
	    {
	       for(int x = 0; x < 15; x++)
	       {
		  cout << (obs[y*15 + x] ? "#" : ".");
	       }
	       cout << endl;
	    }
	    cout << "R: " << reward << endl;
	 }
	 t++;
      }

      if(t < rolloutDepth) //Otherwise the tree is as deep as it can be
      {
	 //Now curNode has at least one untried action
	 //Pick one at random so we can generate a new leaf
	 vector<int> untried;
	 for(unsigned a = 0; a < numActions; a++)
	 {
	    if(tree[curNodeIndex].counts[a] == 0)
	    {
	       untried.push_back(a);
	    }
	 }
	 
	 int action = untried[rand()%untried.size()];
	 int reward = rewardFunction(obs, action);
	 rewards.push_back(reward);
	 actions.push_back(action);

	 if(saveRollout)
	 {
	    rolloutActions[t] = action;
	    rolloutRewards[t] = reward;
	    rolloutObs[t] = obs;
	 }

	 bool endEpisode;
	 bool dummyReward;
	 model[m]->sample(action, obs, dummyReward, endEpisode);
	 if(m+1 < maxModelDepth)
	 {
	    m++;
	 }
	 if(t < rolloutDepth-1)
	 {
	    model[m]->update(action, obs, dummyReward, endEpisode, false);
	 }

	 size_t obsHash = hash_range(obs.begin(), obs.end());
	 tree.push_back(MCTSNode(numActions, curNodeIndex));
	 tree[curNodeIndex].children[action][obsHash] = tree.size()-1;

	 if(printRollouts)
	 {
	    cout << "Leaf A: " << action << endl;
	    for(int y = 0; y < 15; y++)
	    {
	       for(int x = 0; x < 15; x++)
	       {
		  cout << (obs[y*15 + x] ? "#" : ".");
	       }
	       cout << endl;
	    }
	    cout << "R: " << reward << endl;
	 }	 

	 t++;
	 
	 //Now perform a rollout
	 double discount = 1;
	 while(t < rolloutDepth)
	 {
	    int action = rand()%numActions;
	    int reward = rewardFunction(obs, action);
	    
	    if(saveRollout)
	    {
	       rolloutActions[t] = action;
	       rolloutRewards[t] = reward;
	       rolloutObs[t] = obs;
	    }
	    
	    bool endEpisode;
	    bool dummyReward;
	    model[m]->sample(action, obs, dummyReward, endEpisode);
	    if(m+1 < maxModelDepth)
	    {
	       m++;
	    }
	    if(t < rolloutDepth-1)
	    {
	       model[m]->update(action, obs, dummyReward, endEpisode, false);
	    }	    

	    rolloutReturn += discount*reward;
	    discount *= discountFactor;
	    
	    if(printRollouts)
	    {
	       cout << "Rollout A: " << action << endl;
	       for(int y = 0; y < 15; y++)
	       {
		  for(int x = 0; x < 15; x++)
		  {
		     cout << (obs[y*15 + x] ? "#" : ".");
		  }
		  cout << endl;
	       }
	       cout << "R: " << reward << endl;
	    }

	    t++;
	 }
      }

      //Now we back up the return
      int rewardIndex = rewards.size()-1;
      while(curNodeIndex != -1)
      {
	 rolloutReturn = rewards[rewardIndex] + discountFactor*rolloutReturn;
	 int action = actions[rewardIndex];

	 tree[curNodeIndex].totalCount += 1;
	 tree[curNodeIndex].counts[action] += 1;
	 tree[curNodeIndex].summedReturns[action] += rolloutReturn;

	 curNodeIndex = tree[curNodeIndex].parent;

	 rewardIndex--;	 
      }

      if(printRollouts)
      {
	 cout << "Return: " << rolloutReturn << endl;
      }

      for(int m = 0; m < maxModelDepth; m++)
      {
	 model[m]->retrieveState();
      }
   }

   //Break ties randomly
   MCTSNode* root = &(tree[0]);
   double maxVal = -numeric_limits<double>::infinity();
   vector<int> maxActs;
   for(int a = 0; a < numActions; a++)
   {
      double qValue = root->summedReturns[a]/root->counts[a];
      if(fabs(qValue - maxVal) <= 1e-6)
      {
	 maxActs.push_back(a);
      }
      else if(qValue > maxVal)
      {
	 maxVal = qValue;
	 maxActs.clear();
	 maxActs.push_back(a);
      }
   }

   return maxActs[rand()%maxActs.size()];
}

/* Takes a model, discount factor, current observation
   and uses one-ply Monte Carlo to choose an action.
   Optional parameters:
   printRollouts - if true, prints things for debugging.*/
int mcts(SamplingModel<int>* model, double discountFactor, const vector<int>& curObs, bool printRollouts=false)
{
   int numRollouts = 200;
   int rolloutDepth = 15; //Hardcoding these for now. Probably ought to be parameters...
   int numActions = model->getNumActs();
   vector<MCTSNode> tree(1, MCTSNode(numActions, -1));
   for(int rollout = 0; rollout < numRollouts; rollout++)
   {
      if(printRollouts)
      {
	 cout << "Rollout " << rollout << endl;
      }
      model->saveState();

      int t = 0;

      vector<int> obs = curObs;
      vector<int> rewards;
      vector<int> actions;
      int curNodeIndex = 0;
      double rolloutReturn = 0;   

      //Move down the tree until a node is missing a child
      while(t < rolloutDepth && tree[curNodeIndex].totalCount >= numActions)
      {      
	 vector<int> maxActs;
	 double maxScore = -numeric_limits<double>::infinity();
	 for(unsigned a = 0; a < numActions; a++)
	 {
	    double score = tree[curNodeIndex].summedReturns[a]/tree[curNodeIndex].counts[a] + 4*sqrt(log(tree[curNodeIndex].totalCount)/tree[curNodeIndex].counts[a]);
	    if(score-maxScore > 1e-6)
	    {
	       maxActs.clear();
	       maxActs.push_back(a);
	       maxScore = score;
	    }
	    else if(fabs(score-maxScore) <= 1e-6)
	    {
	       maxActs.push_back(a);
	    }	 
	 }
	 int action = maxActs[rand()%maxActs.size()];
	 int reward = rewardFunction(obs, action);
	 rewards.push_back(reward);
	 actions.push_back(action);

	 bool endEpisode;
	 int dummyReward;
	 model->takeAction(action, obs, dummyReward, endEpisode);

	 size_t obsHash = hash_range(obs.begin(), obs.end());
	 int childIndex = tree[curNodeIndex].children[action][obsHash];
	 if(childIndex == 0) //first time seeing this observation, need a new node
	 {
	    tree.push_back(MCTSNode(numActions, curNodeIndex));
	    childIndex = tree.size()-1;
	    tree[curNodeIndex].children[action][obsHash] = childIndex;
	 }

	 curNodeIndex = childIndex;
	 if(printRollouts)
	 {
	    cout << t << ": in tree A: " << action << endl;
	    for(int y = 0; y < 15; y++)
	    {
	       for(int x = 0; x < 15; x++)
	       {
		  cout << (obs[y*15 + x] ? "#" : ".");
	       }
	       cout << endl;
	    }
	    cout << "R: " << reward << endl;
	 }
	 t++;
      }

      if(t < rolloutDepth) //Otherwise the tree is as deep as it can be
      {
	 //Now curNode has at least one untried action
	 //Pick one at random so we can generate a new leaf
	 vector<int> untried;
	 for(unsigned a = 0; a < numActions; a++)
	 {
	    if(tree[curNodeIndex].counts[a] == 0)
	    {
	       untried.push_back(a);
	    }
	 }
	 
	 int action = untried[rand()%untried.size()];
	 int reward = rewardFunction(obs, action);
	 rewards.push_back(reward);
	 actions.push_back(action);

	 bool endEpisode;
	 int dummyReward;
	 model->takeAction(action, obs, dummyReward, endEpisode);

	 size_t obsHash = hash_range(obs.begin(), obs.end());
	 tree.push_back(MCTSNode(numActions, curNodeIndex));
	 tree[curNodeIndex].children[action][obsHash] = tree.size()-1;

	 if(printRollouts)
	 {
	    cout << t << ": leaf A: " << action << endl;
	    for(int y = 0; y < 15; y++)
	    {
	       for(int x = 0; x < 15; x++)
	       {
		  cout << (obs[y*15 + x] ? "#" : ".");
	       }
	       cout << endl;
	    }
	    cout << "R: " << reward << endl;
	 }	 

	 t++;
	 
	 //Now perform a rollout
	 double discount = 1;
	 while(t < rolloutDepth)
	 {
	    int action = rand()%numActions;
	    int reward = rewardFunction(obs, action);
	    
	    bool endEpisode;
	    int dummyReward;
	    model->takeAction(action, obs, dummyReward, endEpisode);

	    rolloutReturn += discount*reward;
	    discount *= discountFactor;
	    
	    if(printRollouts)
	    {
	       cout << t << ": rollout A: " << action << endl;
	       for(int y = 0; y < 15; y++)
	       {
		  for(int x = 0; x < 15; x++)
		  {
		     cout << (obs[y*15 + x] ? "#" : ".");
		  }
		  cout << endl;
	       }
	       cout << "R: " << reward << endl;
	    }

	    t++;
	 }
      }

      //Now we back up the return
      int rewardIndex = rewards.size()-1;
      while(curNodeIndex != -1)
      {
	 rolloutReturn = rewards[rewardIndex] + discountFactor*rolloutReturn;
	 int action = actions[rewardIndex];

	 tree[curNodeIndex].totalCount += 1;
	 tree[curNodeIndex].counts[action] += 1;
	 tree[curNodeIndex].summedReturns[action] += rolloutReturn;

	 curNodeIndex = tree[curNodeIndex].parent;

	 rewardIndex--;
      }

      if(printRollouts)
      {
	 cout << "Return: " << rolloutReturn << endl;
      }

      model->retrieveState();
   }

   //Break ties randomly
   MCTSNode* root = &(tree[0]);
   double maxVal = -numeric_limits<double>::infinity();
   vector<int> maxActs;
   for(int a = 0; a < numActions; a++)
   {
      double qValue = root->summedReturns[a]/root->counts[a];
      //cout << a << ": " << qValue << " " << root->counts[a] << endl;
      if(fabs(qValue - maxVal) <= 1e-6)
      {
	 maxActs.push_back(a);
      }
      else if(qValue > maxVal)
      {
	 maxVal = qValue;
	 maxActs.clear();
	 maxActs.push_back(a);
      }
   }

   return maxActs[rand()%maxActs.size()];
}

/*Evaluates the policy associated with a given model by
  executing it in the world.
  Parameters:
  model - an unrolled learned model to plan with
  world - a perfect model to evaluate in
  discount factor
  policyCache - the theory is for a version of one-ply MC
  generates an action for each state, not a new action
  each time a state is visited. The cache ensures that
  each state is assigned a single action.
  maxD - See onePlyMC
  printRollouts - See onePlyMC*/
double evaluate(const vector<ConvolutionalBinaryCTS*>& model, ShooterModel* world, double discountFactor, unordered_map<size_t, int>& policyCache, int maxD=-1, bool printRollouts=false)
{
   double totalDiscountedReward = 0;
   int numEpisodes = 1;

   int maxModelDepth = maxD;
   if(maxD < 0 || maxD > int(model.size()))
   {
      maxModelDepth = model.size();
   }

   cout << "Evaluating ";
   cout.flush();
   for(int ep = 0; ep < numEpisodes; ep++)
   {
      world->reset();
      for(int m = 0; m < maxModelDepth; m++)
      {
	 model[m]->reset();
      }

      int reward;
      bool endEpisode;
      vector<int> obs;

      world->takeAction(0, obs, reward, endEpisode);
      for(int m = 0; m < 15; m++)
      {
	 model[m]->update(0, obs, reward, false, false);
      }

      double discount = 1.0;

      for(int t = 0; t < 30; t++)
      {
	 size_t hash = hash_range(obs.begin(), obs.end());
	 int action = policyCache[hash];
	 if(!action)
	 {
	    action = mcts(model, discountFactor, obs, maxD, printRollouts);
	    policyCache[hash] = action + 1;
	 }
	 else
	 {
	    action--;
	 }

	 int r = rewardFunction(obs, action);
	 world->takeAction(action, obs, reward, endEpisode);
	 totalDiscountedReward += discount*r;
	 discount *= discountFactor;

	 for(int m = 0; m < maxModelDepth; m++)
	 {
	    model[m]->update(action, obs, 0, false, false);
	 }
      }
   }
   return totalDiscountedReward/numEpisodes;
}

/*Chooses an action according to the exploration policy.
  Type 0: Uniform random
  Type 1: Optimal policy
  Type 2: MCTS with a perfect model*/
int explorationPolicy(ShooterModel* world, vector<int>& curObs, int t, double discountFactor, int numActions, int type)
{
   int a = 0;
   if(type == 0) //Uniform random policy
   {
      a = rand()%numActions;
   }
   else if(type == 2) //One-ply MC with a perfect model
   {
      a = mcts(world, discountFactor, curObs);
   }
   else if(type == 1) //Optimal policy
   {
      a = 2;
      if(t == 1 || t == 7 || t == 13)
      {
	 a = 3;
      }
   }

   return a;
}

int main(int argc, char** argv)
{
   if(argc <= 6)
   {
      cout << "Usage: ./shooterDAggerMCTSUnrolled algorithm explorationType trial numBatches samplesPerBatch movingBullseye [outputFileNote]" << endl;
      cout << "algorithm -- 0: H-DAgger-MCTS, 1: MCTS with perfect model, 2: Uniform random, 3: Optimal policy" << endl;
      cout << "explorationType -- 0: Uniform random, 1: Optimal policy, 2: MCTS with perfect model" << endl;
      cout << "trial -- the trial number (determines the random seed)" << endl;
      cout << "numBatches -- the number of batches to generate in DAgger-based algorithms" << endl;
      cout << "samplesPerBatch -- the number of samples to generate in each batch" << endl;
      cout << "movingBullseye -- 0: bullseyes stay still, 1: bullseyes move" << endl;
      cout << "outputFileNote -- adds the given string to the output filename" << endl;
      exit(1);
   }

   int gamma = 9;
   double discountFactor = double(gamma)/10;

   //0: Hallucinated DAgger-MCTS
   //1: Real model
   //2: Random
   //3: Optimal
   int daggerType = atoi(argv[1]);
   
   //0: Random
   //1: Optimal
   //2: MCTS
   int explorationType = atoi(argv[2]);
   int trial = atoi(argv[3]);
   int numBatches = atoi(argv[4]);
   int samplesPerBatch = atoi(argv[5]);
   int neighborhoodWidth = 7;
   int neighborhoodHeight = 7;
   bool movingSweetSpot = atoi(argv[6]);
   int hDelay = 10;

   int outputNoteIndex = 7;
   string outputNote;
   if(argc > outputNoteIndex)
   {
      outputNote = string(".") + string(argv[outputNoteIndex]);
   }

   //Generate the output file name
   stringstream outSS;
   outSS << "inProgress/shooter";
   if(movingSweetSpot)
   {
      outSS << ".movingSweetSpot";
   }
   outSS << outputNote;
   if(daggerType == 0)
   {
      outSS << ".HDAggerMCTS.unrolled.hDelay" << hDelay;
   }
   else if(daggerType == 1)
   {
      outSS << ".PerfectModelMCTS";
   }
   else if(daggerType == 2)
   {
      outSS << ".Random";
   }
   else if(daggerType == 3)
   {
      outSS << ".Optimal";
   }   

   if(daggerType == 0)
   {
      if(explorationType == 0)
      {
	 outSS << ".randomExplore";
      }
      else if(explorationType == 1)
      {
	 outSS << ".optimalExplore";
      }
      else if(explorationType == 2)
      {
	 outSS << ".mctsExplore";
      }
      outSS << ".nbhd" << neighborhoodWidth << "x" << neighborhoodHeight << ".spb" << samplesPerBatch << ".numBatches" << numBatches;
   }

   outSS << ".t" << trial;

   srand(trial + 1);
   
   ofstream fout(outSS.str().c_str());

   int height = 15;
   int numTargets = 3;
   int numActions = 4;
   ShooterModel* world = new ShooterModel(numTargets, height, movingSweetSpot);
   vector<ConvolutionalBinaryCTS*> model(15);

   for(int m = 0; m < 15; m++)
   {
      model[m] = new ConvolutionalBinaryCTS(height, numTargets*5, neighborhoodHeight, neighborhoodWidth, numActions, 1, trial + 1);
   }

   unordered_map<size_t, int> policyCache;   

   if(daggerType >= 1) //Not really doing DAgger. Just execute one of the benchmark policies and report the results.
   {
      double totalDiscountedReward = 0;
      int numEpisodes = 1;
      vector<int> obs;
      int r;
      bool endEpisode;
      cout << "Evaluating ";
      cout.flush();
      for(int ep = 0; ep < numEpisodes; ep++)
      {
	 world->reset();
	 world->takeAction(0, obs, r, endEpisode);
	 policyCache.clear();
	 
	 double discount = 1.0;
	 for(int t = 0; t < 30; t++)
	 {
	    int action;
	    if(daggerType == 1) //One-ply MC with perfect model
	    {
	       size_t hash = hash_range(obs.begin(), obs.end());
	       action = policyCache[hash];
	       if(!action)
	       {
		  action = mcts(world, discountFactor, obs);
		  policyCache[hash] = action + 1;
	       }
	       else
	       {
		  action--;
	       }
	    }
	    else if(daggerType == 2) //Uniform random policy
	    {
	       action = rand()%numActions;
	    }
	    else //Optimal policy
	    {
	       action = 2;
	       if(t == 1 || t == 7 || t == 13)
	       {
		  action = 3;
	       }
	    }
	    int r = rewardFunction(obs, action);
	    int dummyR;
	    bool endEpisode;
	    world->takeAction(action, obs, dummyR, endEpisode);

	    totalDiscountedReward += discount*r;
	    discount *= discountFactor;
	 }
      }
      cout << totalDiscountedReward/numEpisodes << endl;
      fout << totalDiscountedReward/numEpisodes << endl;
      exit(0);
   }

   //Otherwise...we are actually doing DAgger.

   //One dataset for each sub-model
   vector<vector<tuple<vector<vector<int> >, vector<int>, int, vector<int>, int, bool> > > dataset(15); //obs context, action context, nextAct, nextObs, reward, endEpisode

   //First batch uses only exploration policy
   cout << "Generating Samples";
   cout.flush();
   for(int s = 0; s < samplesPerBatch; s++)
   {
      if((s + 1)%(samplesPerBatch/10) == 0)
      {
	 cout << ".";
	 cout.flush();
      }

      world->reset();

      vector<vector<int> > obsContext(1);
      vector<int> actContext(1);
      int nextAct;
      vector<int> nextObs;
      int reward;
      bool endEpisode;

      //Make a context
      world->takeAction(0, obsContext[0], reward, endEpisode);
      actContext[0] = 0;

      int t = 0;
      while(rand()%10 < gamma)
      {
	 int a = explorationPolicy(world, obsContext[0], t, discountFactor, numActions, explorationType);
	 world->takeAction(a, obsContext[0], reward, endEpisode);
	 actContext[0] = a;
	 t++;
      }

      int a = explorationPolicy(world, obsContext[0], t, discountFactor, numActions, explorationType);
      world->takeAction(a, nextObs, reward, endEpisode);
      nextAct = a;

      for(int m = 0; m < 15; m++)
      {
	 dataset[m].push_back(make_tuple(obsContext, actContext, nextAct, nextObs, 0, false)); 
      }
   }
   
   model[0]->batchUpdate(dataset[0]);

   //Evaluate the first policy
   policyCache.clear();
   double averageDiscountedReward = evaluate(model, world, discountFactor, policyCache, 1);
   cout << "Batch 0 Average Discounted Reward: " << averageDiscountedReward << endl;
   fout << averageDiscountedReward << endl;

   int* rolloutActions = new int[15];
   int* rolloutRewards = new int[15];
   vector<int>* rolloutObservations = new vector<int>[15];

   //Now do the rest of the batches
   for(int b = 1; b < numBatches; b++)
   {
      for(int m = 0; m < 15; m++)
      {
	 dataset[m].clear();
      }

      cout << "Generating Samples";
      cout.flush();
      for(int s = 0; s < samplesPerBatch; s++)
      {
	 if((s + 1)%(samplesPerBatch/10) == 0)
	 {
	    cout << ".";
	    cout.flush();
	 }

	 world->reset();
	 for(int m = 0; m < 15; m++)
	 {
	    model[m]->reset();
	 }
	 vector<vector<int> > obsContext(1);
	 vector<int> actContext(1);
	 vector<int> prevObs;
	 int nextAct;
	 vector<int> nextObs;
	 int reward;
	 bool endEpisode;
	 //Make a context
	 world->takeAction(0, obsContext[0], reward, endEpisode);
	 for(int m = 0; m < 15; m++)
	 {
	    model[m]->update(0, obsContext[0], 0, false, false);
	 }

	 actContext[0] = 0;

	 //Flip a coin
	 int r = rand();
	 r = r%2;
	 if(r) //If heads, use exploration policy to sample a state
	 {
	    int t = 0;
	    //(1 - gamma) probability of termination
	    int term = rand();
	    term = term%10;
	    while(term < gamma)
	    {
	       term = rand();
	       term = term%10;
	       int a = explorationPolicy(world, obsContext[0], t, discountFactor, numActions, explorationType);
	       world->takeAction(a, obsContext[0], reward, endEpisode);

	       for(int m = 0; m < 15; m++)
	       {
		  model[m]->update(a, obsContext[0], 0, false, false);
	       }

	       actContext[0] = a;
	       t++;
	    }

	    //Flip another coin to determine what to do in the last step
	    int coin = rand();
	    coin = coin%2;
	    if(coin) //If coin comes up heads: just use the exploration policy
	    {
	       nextAct = explorationPolicy(world, obsContext[0], t, discountFactor, numActions, explorationType);
	    }
	    else //Otherwise use exploration policy in the last step
	    {
	       size_t hash = hash_range(obsContext[0].begin(), obsContext[0].end());
	       nextAct = policyCache[hash];
	       if(!nextAct)
	       {
		  nextAct = mcts(model, discountFactor, obsContext[0], hDelay > 0 ? b/hDelay+1 : -1);
		  policyCache[hash] = nextAct + 1;
	       }
	       else
	       {
		  nextAct--;
	       }
	    }
	 }
	 else //The first coin (r) was tails: use model policy to sample a state
	 {
	    //(1 - gamma) probability of termination
	    int term = rand();
	    term = term%10;
	    while(term < gamma)
	    {
	       term = rand();
	       term = term%10;
	       size_t hash = hash_range(obsContext[0].begin(), obsContext[0].end());
	       int a = policyCache[hash];
	       if(!a)
	       {
		  a = mcts(model, discountFactor, obsContext[0], hDelay > 0 ? b/hDelay+1 : -1);
		  policyCache[hash] = a + 1;
	       }
	       else
	       {
		  a--;
	       }

	       world->takeAction(a, obsContext[0], reward, endEpisode);
	       for(int m = 0; m < 15; m++)
	       {
		  model[m]->update(a, obsContext[0], 0, false, false);
	       }
	       actContext[0] = a;
	    }

	    size_t hash = hash_range(obsContext[0].begin(), obsContext[0].end());
	    nextAct = policyCache[hash];
	    if(!nextAct)
	    {
	       nextAct = mcts(model, discountFactor, obsContext[0], hDelay > 0 ? b/hDelay+1 : -1);
	       policyCache[hash] = nextAct + 1;
	    }
	    else
	    {
	       nextAct--;
	    }	    
	 }

	 //We have now sampled a state and action -- take the action in that state
	 world->takeAction(nextAct, nextObs, reward, endEpisode);

	 mcts(model, discountFactor, obsContext[0], hDelay > 0 ? b/hDelay+1 : -1, false, nextAct, rolloutActions, rolloutRewards, rolloutObservations); 
	 vector<vector<int> > hObsContext(1);

	 for(int m = 0; m < 15; m++)
	 {
	    if(b >= m*hDelay) //If hallucinating and if we've seen enough batches for this depth, use the hallucinated context
	    {
	       hObsContext[0] = rolloutObservations[m];
	       dataset[m].push_back(make_tuple(hObsContext, actContext, nextAct, nextObs, 0, false));
	    }

	    if(m < 14)
	    {
	       obsContext[0] = nextObs;
	       actContext[0] = nextAct;	       
	       nextAct = rolloutActions[m+1];
	       world->takeAction(nextAct, nextObs, reward, endEpisode);
	    }
	 }
      }

      //Update all the models
      for(int m = 0; m < 15; m++)
      {
	 model[m]->batchUpdate(dataset[m]);
      }

      //Evaluate the policy for this batch
      policyCache.clear();
      double averageDiscountedReward = evaluate(model, world, discountFactor, policyCache, hDelay > 0 ? b/hDelay+1 : -1);
      cout << "Batch " << b << " Discounted Reward: " << averageDiscountedReward << endl;
      fout << averageDiscountedReward << endl;
   }
}
