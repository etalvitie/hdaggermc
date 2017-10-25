#include "ShooterRewardModel.h"

float ShooterRewardModel::getReward(int action, const vector<int>& obs) const
{
   int r = 0;
   if(action == 3)
   {
      r -= 1;
   }

   for(int target = 0; target < 3; target++)
   {
      int bottomLeft = 4*15 + target*5 + 1;
      if(obs[bottomLeft] && !obs[bottomLeft + 1] && obs[bottomLeft + 2] && 
	 !obs[bottomLeft - 15] && obs[bottomLeft - 14] && !obs[bottomLeft - 13])// &&
      {
	 r += 10;
      }

      if(!obs[bottomLeft] && obs[bottomLeft + 1] && !obs[bottomLeft + 2] && 
	 obs[bottomLeft - 15] && !obs[bottomLeft - 14] && obs[bottomLeft - 13])// &&
      {
	 r += 20;
      }
   }
   return r;
}

double ShooterRewardModel::batchMSE(const vector<tuple<vector<int>, int, float, float> >& dataset)
{
   double sse = 0;
   for(unsigned d = 0; d < dataset.size(); d++)
   {
      float r = getReward(dataset[d].get<1>(), dataset[d].get<0>());
      float realR = dataset[d].get<2>();
      sse += (r - realR)*(r - realR)*dataset[d].get<3>();
   }
   return sse/dataset.size();
}

double ShooterRewardModel::batchMSE(const vector<tuple<vector<int>, int, float> >& dataset)
{
   double sse = 0;
   for(unsigned d = 0; d < dataset.size(); d++)
   {
      float r = getReward(dataset[d].get<1>(), dataset[d].get<0>());
      float realR = dataset[d].get<2>();
      sse += (r - realR)*(r - realR);
   }
   return sse/dataset.size();
}
