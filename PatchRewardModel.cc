#include "PatchRewardModel.h"

PatchRewardModel::PatchRewardModel(int numActions, int width, int height, int patchWidth, int patchHeight, float stepSize) : numActions(numActions), width(width), height(height)
{
   this->stepSize = stepSize/(width*height+1);

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

   numExamples = 0;
}

void PatchRewardModel::getActiveFeatures(int action, const vector<int>& obs, vector<int>& indices) const
{
   /*
   cout << endl;
   for(int r = 0; r < height; r++)
   {
      for(int c = 0; c < width; c++)
      {
	 cout << (obs[r*width + c] ? "#" : ".");
      }
      cout << endl;
   }
   */
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

//      cout << index << "(";

      index += p*numPatches + 1;

//      cout << index << ") ";
//      if(p%15 == 14)
//	 cout << endl;

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

double PatchRewardModel::batchUpdate(const vector<tuple<vector<int>, int, float> >& dataset)
{
   float sse = 0;
   int count = 0;

//   vector<unordered_map<int, float> > updates(numActions);
//   vector<int> numDataPoints(numActions, 0);
//   numExamples++;   
   for(unsigned d = 0; d < dataset.size(); d++)
   {
      count++;
//      numExamples++;

      const vector<int>& obs = dataset[d].get<0>();
      int act = dataset[d].get<1>();
      float r = dataset[d].get<2>();

      getActiveFeatures(act, obs, activeFeatures);

      float myR = getRewardFromIndices(act, activeFeatures);

      float error = r - myR;
      sse += error*error;
/*
      if(error != 0)
      {
	 cout << endl;
	 for(int r = 0; r < height; r++)
	 {
	    for(int c = 0; c < width; c++)
	    {
	       cout << (obs[r*width + c] ? "#" : ".");
	    }
	    cout << endl;
	 }
	 cout << "Action: " << act << " R: " << r << " MyR: " << myR << endl;
      }
*/
//      numDataPoints[act]++;
//	 cout << error << " " << 1/(sqrt(numExamples)*(height*width+1)) << endl;
      for(unsigned i = 0; i < activeFeatures.size(); i++)
      {
//	 updates[act][activeFeatures[i]] += error;
	 weights[act][activeFeatures[i]] += stepSize*error;
//	 cout << numExamples << " " << sqrt(numExamples) << endl;
//	 weights[act][activeFeatures[i]] += error/(sqrt(numExamples)*(height*width+1));
//	 cout << weights[act][activeFeatures[i]] << endl;
      }
   }

//   stepSize = 1.0/(sqrt(numExamples)*(height*width + 1));
//   stepSize = 1.0/(sqrt(numExamples));
/*
   for(int a = 0; a < numActions; a++)
   {
      for(unordered_map<int, float>::iterator iter = updates[a].begin(); iter != updates[a].end(); iter++)
      {
	 weights[a][iter->first] += stepSize * iter->second/numDataPoints[a];
      }
   }
*/
   return sse/count;
}

double PatchRewardModel::batchUpdate(const vector<tuple<vector<int>, int, float, float> >& dataset)
{
   float sse = 0;
   int count = 0;

//   vector<unordered_map<int, float> > updates(numActions);
//   vector<int> numDataPoints(numActions, 0);
//   numExamples++;   
   for(unsigned d = 0; d < dataset.size(); d++)
   {
      count++;
//      numExamples++;

      const vector<int>& obs = dataset[d].get<0>();
      int act = dataset[d].get<1>();
      float r = dataset[d].get<2>();

      getActiveFeatures(act, obs, activeFeatures);

      float myR = getRewardFromIndices(act, activeFeatures);

      float error = r - myR;
      sse += error*error*dataset[d].get<3>();
/*
      if(error != 0)
      {
	 cout << endl;
	 for(int r = 0; r < height; r++)
	 {
	    for(int c = 0; c < width; c++)
	    {
	       cout << (obs[r*width + c] ? "#" : ".");
	    }
	    cout << endl;
	 }
	 cout << "Action: " << act << " R: " << r << " MyR: " << myR << endl;
      }
*/
//      numDataPoints[act]++;
//	 cout << error << " " << 1/(sqrt(numExamples)*(height*width+1)) << endl;
      for(unsigned i = 0; i < activeFeatures.size(); i++)
      {
//	 updates[act][activeFeatures[i]] += error;
	 weights[act][activeFeatures[i]] += stepSize*error*dataset[d].get<3>();
//	 cout << numExamples << " " << sqrt(numExamples) << endl;
//	 weights[act][activeFeatures[i]] += error/(sqrt(numExamples)*(height*width+1));
//	 cout << weights[act][activeFeatures[i]] << endl;
      }
   }

//   stepSize = 1.0/(sqrt(numExamples)*(height*width + 1));
//   stepSize = 1.0/(sqrt(numExamples));
/*
   for(int a = 0; a < numActions; a++)
   {
      for(unordered_map<int, float>::iterator iter = updates[a].begin(); iter != updates[a].end(); iter++)
      {
	 weights[a][iter->first] += stepSize * iter->second/numDataPoints[a];
      }
   }
*/
   return sse/count;
}

double PatchRewardModel::batchMSE(const vector<tuple<vector<int>, int, float> >& dataset)
{
   float sse = 0;
   int count = 0;

   if(dataset.size() == 0)
   {
      return 0;
   }

   for(unsigned d = 0; d < dataset.size(); d++)
   {
      count++;

      const vector<int>& obs = dataset[d].get<0>();
      int act = dataset[d].get<1>();
      float r = dataset[d].get<2>();

      getActiveFeatures(act, obs, activeFeatures);

      float myR = getRewardFromIndices(act, activeFeatures);

      float error = r - myR;
      sse += error*error;
   }
   return sse/count;
}

double PatchRewardModel::batchMSE(const vector<tuple<vector<int>, int, float, float> >& dataset)
{
   float sse = 0;
   int count = 0;

   if(dataset.size() == 0)
   {
      return 0;
   }

   for(unsigned d = 0; d < dataset.size(); d++)
   {
      count++;

      const vector<int>& obs = dataset[d].get<0>();
      int act = dataset[d].get<1>();
      float r = dataset[d].get<2>();

      getActiveFeatures(act, obs, activeFeatures);

      float myR = getRewardFromIndices(act, activeFeatures);

      float error = r - myR;
      sse += error*error*dataset[d].get<3>();
   }

   return sse/count;
}
