/********************
Author: Erik Talvitie
********************/

#include "ConvolutionalBinaryCTS.h"
#include "ShooterModel.h"
#include "PatchRewardModel.h"
#include "ShooterRewardModel.h"

#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include <boost/tuple/tuple.hpp>
#include <boost/unordered_map.hpp>
#include <boost/functional/hash.hpp>

using namespace std;
using namespace boost;

/* Takes a model, discount factor, current observation
   and uses one-ply Monte Carlo to choose an action.
   Optional parameters:
   printReturns/printRollouts - if true, prints things for debugging.*/
int onePlyMC(SamplingModel<int>* model, RewardModel* rewardModel, double discountFactor, int rolloutsPerA, int rolloutDepth, const vector<int>& curObs, bool printReturns=false, bool printRollouts=false)
{
   int numActions = model->getNumActs();
   vector<double> returns(numActions, 0);
   for(int a = 0; a < numActions; a++)
   {
      for(int rollout = 0; rollout < rolloutsPerA; rollout++)
      {
	 model->saveState();

	 int action = a;
	 double discount = 1;
	 if(printRollouts)
	 {
	    cout << "Rollout " << rollout << endl;
	 }
	 double rolloutReturn = 0;
	 vector<int> obs = curObs;
	 for(int t = 0; t < rolloutDepth; t++)
	 {
	    float reward = rewardModel->getReward(action, obs);
	    bool endEpisode;
	    int dummyReward;
	    model->takeAction(action, obs, dummyReward, endEpisode);
	    if(printRollouts)
	    {
	       cout << "A: " << action << endl;
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
	    returns[a] += discount*reward;
	    rolloutReturn += discount*reward;
	    discount *= discountFactor;

	    action = rand()%numActions;
	 }
	 if(printRollouts)
	 {
	    cout << "Return: " << rolloutReturn << endl;
	 }
	 model->retrieveState();
      }
   }

   if(printReturns)
   {
      cout << "Returns: ";
      for(int i = 0; i < numActions; i++)
      {
	 cout << returns[i] << " ";
      }
      cout << endl;
   }

   //Break ties randomly
   double maxReturn = returns[0];
   vector<int> maxActs(1, 0);
   for(int i = 1; i < numActions; i++)
   {
      if(abs(returns[i] - maxReturn) < 1e-6)
      {
	 maxActs.push_back(i);
      }
      else if(returns[i] > maxReturn)
      {
	 maxReturn = returns[i];
	 maxActs.clear();
	 maxActs.push_back(i);
      }
   }
   int choice = rand();
   choice = choice%maxActs.size();
   return maxActs[choice];
}
      
/*Evaluates the policy associated with a given model by
  executing it in the world.
  Parameters:
  model - a learned model to plan with
  world - a perfect model to evaluate in
  discount factor
  policyCache - the theory is for a version of one-ply MC
  generates an action for each state, not a new action
  each time a state is visited. The cache ensures that
  each state is assigned a single action.
  printRollouts - See onePlyMC*/
tuple<double, double, double> evaluate(ConvolutionalBinaryCTS* model, RewardModel* rewardModel, ShooterModel* world, ShooterRewardModel* worldReward, double discountFactor, int rolloutsPerA, int rolloutDepth, unordered_map<size_t, int>& policyCache, bool printRollouts=false)
{
   double totalDiscountedReward = 0;
   double ll = 0;
   double rewardSSE = 0;
   int count = 0;

   int numEpisodes = 1;

   cout << "Evaluating ";
   cout.flush();
   for(int ep = 0; ep < numEpisodes; ep++)
   {
      world->reset();
      model->reset();

      int reward;
      bool endEpisode;
      vector<int> obs;

      world->takeAction(0, obs, reward, endEpisode);
      model->update(0, obs, reward, false, false);

      double discount = 1.0;

      for(int t = 0; t < 30; t++)
      {

	 size_t hash = hash_range(obs.begin(), obs.end());
	 int action = policyCache[hash];
	 if(!action)
	 {
	    action = onePlyMC(model, rewardModel, discountFactor, rolloutsPerA, rolloutDepth, obs, false, printRollouts);
	    policyCache[hash] = action + 1;
	 }
	 else
	 {
	    action--;
	 }

	 float r = worldReward->getReward(action, obs);
	 float predictedR = rewardModel->getReward(action, obs);
	 rewardSSE += (r - predictedR)*(r - predictedR);
	 count++;

	 world->takeAction(action, obs, reward, endEpisode);
	 totalDiscountedReward += discount*r;
	 discount *= discountFactor;

	 ll += log(model->predict(action, obs));
	 model->update(action, obs, 0, false, false);
      }
   }
   return make_tuple(totalDiscountedReward/numEpisodes, ll, rewardSSE/count);
}

/*Chooses an action according to the exploration policy.
  Type 0: Uniform random
  Type 1: Optimal policy
  Type 2: One-ply MC with a perfect model*/
int explorationPolicy(ShooterModel* world, ShooterRewardModel* worldReward, vector<int>& curObs, int t, double discountFactor, int rolloutsPerA, int rolloutDepth, int numActions, int type)
{
   int a = 0;
   if(type == 0) //Uniform random policy
   {
      a = rand()%numActions;
   }
   else if(type == 2) //One-ply MC with a perfect model
   {
      a = onePlyMC(world, worldReward, discountFactor, rolloutsPerA, rolloutDepth, curObs);
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
   if(argc <= 9)
   {
      cout << "Usage: ./shooterDAggerUnrolled algorithm explorationType trial numBatches samplesPerBatch movingBullseye maxHDepth [outputFileNote]" << endl;
      cout << "algorithm -- 0: DAgger, 1: DAgger-MC, 2: H-DAgger-MC, 3: One-ply MC with perfect model, 4: Uniform random, 5: Optimal policy" << endl;
      cout << "explorationType -- 0: Uniform random, 1: Optimal policy, 2: One-ply MC with perfect model" << endl;
      cout << "rewardType -- 0: Perfect reward, 1: Learned from real states, 2: learned from hallucinated states" << endl;
      cout << "rewardStepSize -- the step size used in the reward model's gradient descent" << endl;
      cout << "rolloutsPerAction -- the number of rollouts to perform for each action during planning" << endl;
      cout << "rolloutDepth -- the depth of each rollout during planning" << endl;
      cout << "trial -- the trial number (determines the random seed)" << endl;
      cout << "numBatches -- the number of batches to generate in DAgger-based algorithms" << endl;
      cout << "samplesPerBatch -- the number of samples to generate in each batch" << endl;
      cout << "movingBullseye -- 0: bullseyes stay still, 1: bullseyes move" << endl;
      cout << "maxHDepth -- maximum hallucinated rollout depth during training" << endl;
      cout << "outputFileNote -- adds the given string to the output filename" << endl;
      exit(1);
   }

   int gamma = 9;
   double discountFactor = double(gamma)/10;

   //0: DAgger
   //1: DAgger-MC
   //2: Hallucinated DAgger-MC
   //3: Real model
   //4: Random
   //5: Optimal
   int daggerType = atoi(argv[1]);
   //0: Random
   //1: Optimal
   //2: One-ply MC
   int explorationType = atoi(argv[2]);
   //0: Perfect
   //1: Learned from real states
   //2: Learned from hallucinated states
   int rewardType = atoi(argv[3]);
   double rewardStepSize = atof(argv[4]);
   int rolloutsPerA = atoi(argv[5]);
   int rolloutDepth = atoi(argv[6]);
   int trial = atoi(argv[7]);
   int numBatches = atoi(argv[8]);
   int samplesPerBatch = atoi(argv[9]);
   int neighborhoodWidth = 7;
   int neighborhoodHeight = 7;
   bool movingSweetSpot = atoi(argv[10]);
   int hDelay = 10;
   int maxH = atoi(argv[11]);

   int outputNoteIndex = 12;
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
      outSS << ".DAgger";
   }
   else if(daggerType == 1)
   {
      outSS << ".DAggerMC.undiscounted";
   }
   else if(daggerType == 2)
   {
      outSS << ".HDAggerMC.undiscounted.hDelay" << hDelay << ".maxH" << maxH;
   }
   else if(daggerType == 3)
   {
      outSS << ".PerfectModel";
   }
   else if(daggerType == 4)
   {
      outSS << ".Random";
   }
   else if(daggerType == 5)
   {
      outSS << ".Optimal";
   }   

   if(daggerType < 4)
   {
      outSS << ".rolloutsPerA" << rolloutsPerA << ".rolloutD" << rolloutDepth;
   }

   if(daggerType < 3)
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
	 outSS << ".mcExplore";
      }

      if(rewardType == 1)
      {
	 outSS << ".realStateReward";
      }
      else if(rewardType == 2)
      {
	 outSS << ".hallucinatedReward";
      }

      if(rewardType > 0)
      {
	 outSS << ".stepSize" << rewardStepSize;
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
   ShooterRewardModel* worldReward = new ShooterRewardModel();

   ConvolutionalBinaryCTS* model = new ConvolutionalBinaryCTS(height, numTargets*5, neighborhoodHeight, neighborhoodWidth, numActions, 1, trial + 1);

   RewardModel* rewardModel;
   if(rewardType > 0)
   {
      rewardModel = new PatchRewardModel(numActions, numTargets*5, height, 3, 3, rewardStepSize);
   }
   else
   {
      rewardModel = new ShooterRewardModel();
   }

   unordered_map<size_t, int> policyCache;   

   if(daggerType >= 3) //Not really doing DAgger. Just execute one of the benchmark policies and report the results.
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
	    if(daggerType == 3) //One-ply MC with perfect model
	    {
	       size_t hash = hash_range(obs.begin(), obs.end());
	       action = policyCache[hash];
	       if(!action)
	       {
		  action = onePlyMC(world, worldReward, discountFactor, rolloutsPerA, rolloutDepth, obs);
		  policyCache[hash] = action + 1;
	       }
	       else
	       {
		  action--;
	       }
	    }
	    else if(daggerType == 4) //Uniform random policy
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
	    int r = worldReward->getReward(action, obs);
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

   vector<tuple<vector<vector<int> >, vector<int>, int, vector<int>, int, bool> > dataset; //obs context, action context, nextAct, nextObs, reward, endEpisode
   vector<tuple<vector<vector<int> >, vector<int>, int, vector<int>, int, bool> > hdataset; //obs context, action context, nextAct, nextObs, reward, endEpisode

   vector<tuple<vector<int>, int, float, float> > rDataset; //obs, action, reward, weight
   vector<tuple<vector<int>, int, float, float> > hrDataset; //obs, action, reward, weight
   
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
      int dummyReward;
      bool endEpisode;
      float reward;

      //Make a context
      world->takeAction(0, obsContext[0], dummyReward, endEpisode);
      actContext[0] = 0;

      int t = 0;
      while(rand()%10 < gamma)
      {
	 int a = explorationPolicy(world, worldReward, obsContext[0], t, discountFactor, rolloutsPerA, rolloutDepth, numActions, explorationType);
	 world->takeAction(a, obsContext[0], dummyReward, endEpisode);
	 actContext[0] = a;
	 t++;
      }

      int a = explorationPolicy(world, worldReward, obsContext[0], t, discountFactor, rolloutsPerA, rolloutDepth, numActions, explorationType);
      reward = worldReward->getReward(a, obsContext[0]);
      world->takeAction(a, nextObs, dummyReward, endEpisode);
      nextAct = a;

      dataset.push_back(make_tuple(obsContext, actContext, nextAct, nextObs, 0, false));
      rDataset.push_back(make_tuple(obsContext[0], nextAct, reward, 1));
   }

   //Update the model
   model->batchUpdate(dataset);
   rewardModel->batchUpdate(rDataset);

   double ll = model->batchLL(dataset);
   double hll = ll;
   double mse = rewardModel->batchMSE(rDataset);

   //Evaluate the first policy
   policyCache.clear();
   tuple<double, double, double> results = evaluate(model, rewardModel, world, worldReward, discountFactor, rolloutsPerA, rolloutDepth, policyCache);//, true);
   cout << "Batch 0 Discounted Reward: " << results.get<0>() << endl;
   fout << results.get<0>() << " " << results.get<1>() << " " << results.get<2>() << " " << ll << " " << ll << " " << mse << " " << mse << endl;

   //Now do the rest of the batches
   for(int b = 1; b < numBatches; b++)
   {
      dataset.clear();
      hdataset.clear();
      rDataset.clear();
      hrDataset.clear();
      
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
	 model->reset();
	 vector<vector<int> > obsContext(1);
	 vector<int> actContext(1);
	 vector<int> prevObs;
	 int nextAct;
	 vector<int> nextObs;
	 int dummyReward;
	 bool endEpisode;
	 //Make a context
	 world->takeAction(0, obsContext[0], dummyReward, endEpisode);
	 model->update(0, obsContext[0], 0, false, false);

	 actContext[0] = 0;

	 //Flip a coin
	 int r = rand();
	 r = r%2;
	 if(rand()%2) //If heads, use exploration policy to sample a state
	 {
	    int t = 0;
	    //(1 - gamma) probability of termination
	    int term = rand();
	    term = term%10;
	    while(term < gamma)
	    {
	       term = rand();
	       term = term%10;
	       int a = explorationPolicy(world, worldReward, obsContext[0], t, discountFactor, rolloutsPerA, rolloutDepth, numActions, explorationType);
	       world->takeAction(a, obsContext[0], dummyReward, endEpisode);
	       if(daggerType > 0)
	       {
		  model->update(a, obsContext[0], 0, false, false);
	       }
	       actContext[0] = a;
	       t++;
	    }

	    //Flip another coin to determine what to do in the last step
	    int coin = rand();
	    coin = coin%2;
	    if(daggerType == 0 || coin) //If doing regular DAgger, or if coin comes up heads: just use the exploration policy
	    {
	       nextAct = explorationPolicy(world, worldReward, obsContext[0], t, discountFactor, rolloutsPerA, rolloutDepth, numActions, explorationType);
	    }
	    else //Otherwise use exploration policy in the last step
	    {
	       size_t hash = hash_range(obsContext[0].begin(), obsContext[0].end());
	       nextAct = policyCache[hash];
	       if(!nextAct)
	       {
		  nextAct = onePlyMC(model, rewardModel, discountFactor, rolloutsPerA, rolloutDepth, obsContext[0]);
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
		  a = onePlyMC(model, rewardModel, discountFactor, rolloutsPerA, rolloutDepth, obsContext[0]);
		  policyCache[hash] = a + 1;
	       }
	       else
	       {
		  a--;
	       }

	       world->takeAction(a, obsContext[0], dummyReward, endEpisode);
	       model->update(a, obsContext[0], 0, false, false);
	       actContext[0] = a;
	    }

	    size_t hash = hash_range(obsContext[0].begin(), obsContext[0].end());
	    nextAct = policyCache[hash];
	    if(!nextAct)
	    {
	       nextAct = onePlyMC(model, rewardModel, discountFactor, rolloutsPerA, rolloutDepth, obsContext[0]);
	       policyCache[hash] = nextAct + 1;
	    }
	    else
	    {
	       nextAct--;
	    }	    
	 }

	 //We have now sampled a state and action -- take the action in that state
	 world->takeAction(nextAct, nextObs, dummyReward, endEpisode);
	 float reward = worldReward->getReward(nextAct, obsContext[0]);

	 if(daggerType > 0) //The hallucinated context starts out the same as the regular context
	 {
	    vector<vector<int> > hObsContext;
	    int hReward;
	    bool hEnd;
	    hObsContext = obsContext;
	    
	    for(int h = 0; h < rolloutDepth; h++)
	    {
	       dataset.push_back(make_tuple(obsContext, actContext, nextAct, nextObs, 0, false));
	       if(b >= h*hDelay && h <= maxH)  //If hallucinating and if we've seen enough batches for this depth, and if we're not past the maximum depth, use the hallucinated context
	       {
		  hdataset.push_back(make_tuple(hObsContext, actContext, nextAct, nextObs, 0, false));
	       }

	       rDataset.push_back(make_tuple(obsContext[0], nextAct, reward, pow(discountFactor, h)));
	       if(b > h*hDelay || b > maxH*hDelay)
	       {
		  hrDataset.push_back(make_tuple(hObsContext[0], nextAct, reward, pow(discountFactor, h)));	  
	       }
	       
	       model->takeAction(nextAct, hObsContext[0], hReward, hEnd);
	       
	       obsContext[0] = nextObs;
	       actContext[0] = nextAct;	       
	       nextAct = rand();
	       nextAct = nextAct%numActions;
	       world->takeAction(nextAct, nextObs, dummyReward, endEpisode);
	       reward = worldReward->getReward(nextAct, obsContext[0]);
	    }
	 }
	 else
	 {
	    dataset.push_back(make_tuple(obsContext, actContext, nextAct, nextObs, 0, false));
	    rDataset.push_back(make_tuple(obsContext[0], nextAct, reward, 1));
	 }
      }

      if(rewardType < 2)
      {
	 rewardModel->batchUpdate(rDataset);
      }
      else
      {
	 rewardModel->batchUpdate(hrDataset);
      }

      double mse = rewardModel->batchMSE(rDataset);
      double hmse = rewardModel->batchMSE(hrDataset);

      //Evaluate the policy for this batch
      policyCache.clear();
      tuple<double, double, double> results = evaluate(model, rewardModel, world, worldReward, discountFactor, rolloutsPerA, rolloutDepth, policyCache);
      cout << "Batch " << b << " Discounted Reward: " << results.get<0>() << endl;
      fout << results.get<0>() << " " << results.get<1>() << " " << results.get<2>() << " " << ll << " " << hll << " " << mse << " " << hmse << endl;

      //Update the model
      if(daggerType < 2)
      {
	 model->batchUpdate(dataset);
      }
      else
      {
	 model->batchUpdate(hdataset);
      }

      ll = model->batchLL(dataset);
      hll = model->batchLL(hdataset);
   }
}
