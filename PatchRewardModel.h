/********************
Author: Erik Talvitie
********************/

#ifndef PATCH_REWARD_MODEL_H
#define PATCH_REWARD_MODEL_H

#include "RewardModel.h"

#include <boost/unordered_map.hpp>
#include <boost/tuple/tuple.hpp>
#include <vector>
#include <iostream>

using namespace std;
using namespace boost;

/* Represents a reward model over binary images that is a linear function of binary features.
   Each feature represents the presense/absence of a particular configuration of pixels at a 
   particular position.
 */
class PatchRewardModel : public RewardModel
{
  protected:
   int numActions;
   int width;
   int height;
   float stepSize;

   int numExamples;
   
   int numPatches;
   vector<unordered_map<int, float> > weights;
   vector<vector<int> > neighborhoods;

   mutable vector<int> activeFeatures;
   
   virtual void getActiveFeatures(int action, const vector<int>& obs, vector<int>& indices) const;
   virtual float getRewardFromIndices(int action, const vector<int>& activeIndices) const;
   
  public:
   //numActions: the number of actions
   //width: the width of the input images (in pixels)
   //height: the height of the input images
   //patchWidth: the width of the patches that define each feature
   //patchHeight: the height of the patches that define each feature
   //stepSize: the step-size parameter to use in stochastic gradient descent
   PatchRewardModel(int numActions, int width, int height, int patchWidth, int patchHeight, float stepSize);

   virtual float getReward(int action, const vector<int>& obs) const;

   //The elements of the tuple are:
   //The input features
   //The action
   //The reward that was observed
   virtual double batchUpdate(const vector<tuple<vector<int>, int, float> >& dataset);
   //The additional element to the tuple is:
   //Weight of this example (for weighted regression)
   virtual double batchUpdate(const vector<tuple<vector<int>, int, float, float> >& dataset);

   virtual double batchMSE(const vector<tuple<vector<int>, int, float, float> >& dataset);
   virtual double batchMSE(const vector<tuple<vector<int>, int, float> >& dataset);
};

#endif
