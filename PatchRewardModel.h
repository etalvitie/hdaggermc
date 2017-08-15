#ifndef PATCH_REWARD_MODEL_H
#define PATCH_REWARD_MODEL_H

#include "RewardModel.h"

#include <boost/unordered_map.hpp>
#include <boost/tuple/tuple.hpp>
#include <vector>
#include <iostream>

using namespace std;
using namespace boost;

class PatchRewardModel : public RewardModel
{
  protected:
   int numActions;
   int width;
   int height;
   float stepSize;
   
   int numPatches;
   vector<unordered_map<int, float> > weights;
   vector<vector<int> > neighborhoods;

   mutable vector<int> activeFeatures;
   
   virtual void getActiveFeatures(int action, const vector<int>& obs, vector<int>& indices) const;
   virtual float getRewardFromIndices(int action, const vector<int>& activeIndices) const;
   
  public:
   PatchRewardModel(int numActions, int width, int height, int patchWidth, int patchHeight, float stepSize);

   virtual float getReward(int action, const vector<int>& obs) const;

   virtual void batchUpdate(const vector<tuple<vector<int>, int, float> >& dataset);
};

#endif
