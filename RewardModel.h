/********************
Author: Erik Talvitie
********************/

#ifndef REWARD_MODEL_H
#define REWARD_MODEL_H

#include <boost/unordered_map.hpp>
#include <boost/tuple/tuple.hpp>
#include <vector>
#include <iostream>

using namespace std;
using namespace boost;

class RewardModel
{
  public:
   virtual float getReward(int action, const vector<int>& obs) const = 0;

   //The elements of the tuple are:
   //The input features
   //The action
   //The reward that was observed
   virtual double batchUpdate(const vector<tuple<vector<int>, int, float> >& dataset) = 0;
   //The additional element to the tuple is:
   //Weight of this example (for weighted regression)
   virtual double batchUpdate(const vector<tuple<vector<int>, int, float, float> >& dataset) = 0;

   virtual double batchMSE(const vector<tuple<vector<int>, int, float, float> >& dataset) = 0;
   virtual double batchMSE(const vector<tuple<vector<int>, int, float> >& dataset) = 0;
};

#endif
