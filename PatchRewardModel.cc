#include "PatchRewardModel.h"

PatchRewardModel::PatchRewardModel(int numActions, int width, int height, int patchWidth, int patchHeight, float stepSize) : numActions(numActions), width(width), height(height), stepSize(stepSize)
{
   numPatches = pow(3, patchWidth*patchHeight);   
   
   weights.resize(numActions);
   
   for(int r = 0; r < height; r++)
   {
      for(int c = 0; c < width; c++)
      {
	 vector<int> nbhd;
	 for(int nr = r - patchHeight/2; nr <= r + (patchHeight-1)/2; nr++)
	 {
	    for(int nc = c - patchWidth/2; nc <= c + (patchWidth-1)/2; nc++)
	    {
	       if(nr < 0 || nr >= height || nc < 0 || nc >= width)
	       {
		  nbhd.push_back(-1);
	       }
	       else
	       {
		  nbhd.push_back(nr*width + nc);
	       }
	    }
	 }
	 neighborhoods.push_back(nbhd);
      }
   }   
}

void PatchRewardModel::getActiveFeatures(int action, const vector<int>& obs, vector<int>& indices) const
{
   indices.clear();

   indices.push_back(0);
   
   for(int p = 0; p < width*height; p++)
   {
      const vector<int>& nbhd = neighborhoods[p];

      int index = 0;
      for(unsigned n = 0; n < nbhd.size(); n++)
      {
	 index *= 3;
	 if(nbhd[n] < 0)
	 {
	    index += 2;
	 }
	 else
	 {
	    index += obs[nbhd[n]];
	 }
      }

      index += p*numPatches + 1;
      indices.push_back(index);
   }
}

float PatchRewardModel::getReward(int action, const vector<int>& obs) const
{
   getActiveFeatures(action, obs, activeFeatures);
   return getRewardFromIndices(action, activeFeatures);
}

float PatchRewardModel::getRewardFromIndices(int action, const vector<int>& activeIndices) const
{
   float r = 0;
   
   for(unsigned i = 0; i < activeIndices.size(); i++)
   {
      unordered_map<int, float>::const_iterator iter = weights[action].find(activeIndices[i]);
      if(iter != weights[action].end())
      {
	 r += iter->second;
      }
   }

   return r;
}

void PatchRewardModel::batchUpdate(const vector<tuple<vector<int>, int, float> >& dataset)
{
   vector<unordered_map<int, float> > updates(numActions);
   vector<int> numDataPoints(numActions, 0);
   
   for(unsigned d = 0; d < dataset.size(); d++)
   {
      const vector<int>& obs = dataset[d].get<0>();
      int act = dataset[d].get<1>();
      float r = dataset[d].get<2>();

      getActiveFeatures(act, obs, activeFeatures);

      float myR = getRewardFromIndices(act, activeFeatures);

      float error = r - myR;

      numDataPoints[act]++;
      for(unsigned i = 0; i < activeFeatures.size(); i++)
      {
	 int index = activeFeatures[i];
	 updates[act][index] += error;
      }
   }

   for(int a = 0; a < numActions; a++)
   {
      for(unordered_map<int, float>::iterator iter = updates[a].begin(); iter != updates[a].end(); iter++)
      {
	 weights[a][iter->first] += stepSize * iter->second/numDataPoints[a];
      }
   }
}
