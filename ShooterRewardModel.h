#ifndef SHOOTER_REWARD_MODEL_H
#define SHOOTER_REWARD_MODEL_H

#include "RewardModel.h"

#include <boost/unordered_map.hpp>
#include <boost/tuple/tuple.hpp>
#include <vector>
#include <iostream>

using namespace std;
using namespace boost;

class ShooterRewardModel : public RewardModel
{
  public:
   virtual float getReward(int action, const vector<int>& obs) const;

   virtual void batchUpdate(const vector<tuple<vector<int>, int, float> >& dataset){};
};

#endif
