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

   virtual void batchUpdate(const vector<tuple<vector<int>, int, float> >& dataset) = 0;
};

#endif
