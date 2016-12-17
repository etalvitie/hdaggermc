/********************
Author: Erik Talvitie
Date: 2016
********************/

#include "ConvolutionalBinaryCTS.h"
#include <fstream>
#include <functional>

ConvolutionalBinaryCTS::ConvolutionalBinaryCTS(int width, int height, int neighborhoodWidth, int neighborhoodHeight, int numActions, int order, int seed) :
   SamplingModel<int>(numActions, width*height),
   width(width),
   height(height),
   neighborhoodWidth(neighborhoodWidth),
   neighborhoodHeight(neighborhoodHeight),
   numColors(2),
   order(order),
   rng(seed),
   internal_uniform(rng),
   uniform(internal_uniform),
   savedHistoryLength(0),
   actHistory(1),
   obsHistory(1),
   rHistory(1),
   endHistory(1),
   offsetToEncodedPos(neighborhoodWidth, vector<int>(neighborhoodHeight))
{
   init(neighborhoodWidth, neighborhoodHeight, numActions, numColors);

   int contextSize = order*(bitsPerPixel*neighborhoodWidth*neighborhoodHeight + bitsPerAction);
   ct = new SwitchingTree(contextSize);

   int globalContextSize = order*((bitsPerPixel - 1)*width*height + bitsPerAction);
   rct = new SwitchingTree(globalContextSize);
   ect = new SwitchingTree(globalContextSize);
}

ConvolutionalBinaryCTS::ConvolutionalBinaryCTS(int width, int height, int neighborhoodWidth, int neighborhoodHeight, int numActions, int order, randgen_t& uniform) :
   SamplingModel<int>(numActions, width*height),
   width(width),
   height(height),
   neighborhoodWidth(neighborhoodWidth),
   neighborhoodHeight(neighborhoodHeight),
   numColors(2),
   order(order),
   internal_uniform(rng),
   uniform(uniform),
   savedHistoryLength(0),
   actHistory(1),
   obsHistory(1),
   rHistory(1),
   endHistory(1),
   offsetToEncodedPos(neighborhoodWidth, vector<int>(neighborhoodHeight))
{
   init(neighborhoodWidth, neighborhoodHeight, numActions, numColors);

   int contextSize = order*(bitsPerPixel*neighborhoodWidth*neighborhoodHeight + bitsPerAction);
   ct = new SwitchingTree(contextSize);

   int globalContextSize = order*((bitsPerPixel - 1)*width*height + bitsPerAction);
   rct = new SwitchingTree(globalContextSize);
   ect = new SwitchingTree(globalContextSize);
}

void ConvolutionalBinaryCTS::init(int neighborhoodWidth, int neighborhoodHeight, int numActions, int numColors)
{
   bitsPerAction = 0;
   for(int i = 1; i < numActions; i *= 2)
   {
      bitsPerAction++;
   }

   bitsPerPixel = 2;

   vector<pair<int, pair<int, int> > > distAndOffset;
   for(int xOff = 0; xOff < neighborhoodWidth; xOff++)
   {
      for(int yOff = 0; yOff < neighborhoodHeight; yOff++)
      {
	 distAndOffset.push_back(make_pair(abs(xOff - neighborhoodWidth/2) + abs(yOff - neighborhoodHeight/2), make_pair(xOff, yOff)));
      }
   }
   sort(distAndOffset.begin(), distAndOffset.end(), greater<pair<int, pair<int, int> > >());

   for(unsigned i = 0; i < distAndOffset.size(); i++)
   {
      offsetToEncodedPos[distAndOffset[i].second.first][distAndOffset[i].second.second] = i;
   } 
}

ConvolutionalBinaryCTS::~ConvolutionalBinaryCTS()
{
   delete ct;
   delete rct;
   delete ect;
}

void ConvolutionalBinaryCTS::encode(const vector<int>& obs, int pos, vector<bit_t>& encoded) const
{
   return encode(obs, pos/height, pos%height, encoded);
}

void ConvolutionalBinaryCTS::encode(const vector<int>& obs, int x, int y, vector<bit_t>& encoded) const
{
   encoded.resize(bitsPerPixel*neighborhoodWidth*neighborhoodHeight);

   for(int xOff = 0; xOff < neighborhoodWidth; xOff++)
   {
      for(int yOff = 0; yOff < neighborhoodHeight; yOff++)
      {
	 int encodedPos = offsetToEncodedPos[xOff][yOff];
	 
	 int actualX = x + xOff - neighborhoodWidth/2;
	 int actualY = y + yOff - neighborhoodHeight/2;

	 int pix;
	 if(actualX >= 0 && actualX < width && actualY >= 0 && actualY < height)
	 {
	    pix = obs[actualX*height + actualY];
	    encoded[bitsPerPixel*encodedPos + bitsPerPixel - 1] = true;
	 }
	 else
	 {
	    pix = 0;
	    encoded[bitsPerPixel*encodedPos + bitsPerPixel - 1] = false;
	 }

	 encoded[bitsPerPixel*encodedPos] = pix ? true : false;
      }
   }
}

void ConvolutionalBinaryCTS::encode(const vector<int>& obs, vector<bit_t>& encoded) const
{
   encoded.resize((bitsPerPixel - 1)*width*height);

   for(int p = 0; p < this->numDim; p++)
   {
      encoded[p*(bitsPerPixel - 1)] = obs[this->numDim - 1 - p] ? true : false;
   }
}

void ConvolutionalBinaryCTS::encode(int act, vector<bit_t>& encoded) const
{
   encoded.resize(bitsPerAction);

   for(int i = 0; i < bitsPerAction; i++)
   {
      encoded[i] = (act & (1 << (bitsPerAction - 1 - i))) ? true : false;
   }
}

void ConvolutionalBinaryCTS::setUpContext(int pos, int traj, int step) const
{
   ct->resetHistory();
   
   //Set up the context for this position
   int contextStart = max(step - order, 0);
   vector<bit_t> act(bitsPerAction);
   vector<bit_t> obs(bitsPerPixel*neighborhoodWidth*neighborhoodHeight);
   for(int t = contextStart; t < step; t++)
   {
      encode(actHistory[traj][t], act);
      ct->updateHistory(act);
      encode(obsHistory[traj][t], pos, obs);
      ct->updateHistory(obs);
   }
}

void ConvolutionalBinaryCTS::setUpContext(int traj, int step) const
{
   rct->resetHistory();
   ect->resetHistory();
   
   //Set up the context for this position
   int contextStart = max(step - order + 1, 0);
   vector<bit_t> act(bitsPerAction);
   vector<bit_t> obs((bitsPerPixel - 1)*width*height);
   for(int t = contextStart; t < step; t++)
   {
      encode(actHistory[traj][t], act);
      rct->updateHistory(act);
      ect->updateHistory(act);
      encode(obsHistory[traj][t], obs);
      rct->updateHistory(obs);
      ect->updateHistory(obs);
   }
}

void ConvolutionalBinaryCTS::update(int act, const vector<int>& obs, int reward, bool endTraj)
{
   update(act, obs, reward, endTraj, true);
}

void ConvolutionalBinaryCTS::update(int act, const vector<int>& obs, bool reward, bool endTraj, bool learn)
{
   actHistory.back().push_back(act);
   obsHistory.back().push_back(obs);
   rHistory.back().push_back(reward);
   endHistory.back().push_back(endTraj);
   if(learn)
   {
      updateActObs(actHistory.size() - 1, actHistory.back().size() - 1, 1);
      updateREnd(actHistory.size() - 1, actHistory.back().size() - 1, 1);
   }
}

void ConvolutionalBinaryCTS::batchUpdate(const vector<tuple<vector<vector<int> >, vector<int>, int, vector<int>, int, bool> >& dataset)
{
   int curLength = actHistory.back().size();

   for(unsigned d = 0; d < dataset.size(); d++)
   {
      const vector<vector<int> >& obsContext = dataset[d].get<0>();
      const vector<int>& actContext = dataset[d].get<1>();
      int nextAct = dataset[d].get<2>();
      const vector<int>& nextObs = dataset[d].get<3>();
      int reward = dataset[d].get<4>();
      bool end = dataset[d].get<5>();

      //Assumes actContext and obsContext are same length...
      for(unsigned i = 0; i < actContext.size(); i++)
      {
	 actHistory.back().push_back(actContext[i]);
	 obsHistory.back().push_back(obsContext[i]);
      }
      actHistory.back().push_back(nextAct);
      obsHistory.back().push_back(nextObs);

      //Dummy reward and end values...
      for(unsigned i = 0; i < actContext.size(); i++)
      {
	 rHistory.back().push_back(0);
	 endHistory.back().push_back(false);
      }
      rHistory.back().push_back(reward);
      endHistory.back().push_back(end);

      //Perform the updates
      updateActObs(actHistory.size() - 1, actHistory.back().size() - 1, 1);
      updateREnd(actHistory.size() - 1, actHistory.back().size() - 1, 1);
      
      //Undo!
      actHistory.back().resize(curLength);
      obsHistory.back().resize(curLength);
      rHistory.back().resize(curLength);
      endHistory.back().resize(curLength);
   }
}

void ConvolutionalBinaryCTS::reset()
{
   actHistory.push_back(vector<int>());
   obsHistory.push_back(vector<vector<int> >());
   rHistory.push_back(vector<bool>());
   endHistory.push_back(vector<bool>());
}

void ConvolutionalBinaryCTS::updateActObs(int traj, int step, int numUpdates)
{
   vector<bit_t> act(bitsPerAction);
   vector<bit_t> obs(bitsPerPixel*neighborhoodWidth*neighborhoodHeight);
   for(int p = 0; p < width*height; p++)
   {
      setUpContext(p, traj, step);
      
      //Now do the updates      
      for(int i = 0; i < numUpdates; i++)
      {
	 encode(actHistory[traj][step + i], act);
	 ct->updateHistory(act);	    
	 encode(obsHistory[traj][step + i], p, obs);
	 ct->update(obs[bitsPerPixel*(neighborhoodWidth*neighborhoodHeight - 1)]);
      }
   }      
}

void ConvolutionalBinaryCTS::updateREnd(int traj, int step, int numUpdates)
{
   vector<bit_t> act(bitsPerAction);
   vector<bit_t> globalObs((bitsPerPixel - 1)*width*height);
   setUpContext(traj, step);
   for(int i = 0; i < numUpdates; i++)
   {
      encode(actHistory[traj][step + i], act);
      rct->updateHistory(act);
      ect->updateHistory(act);

      encode(obsHistory[traj][step + i], globalObs);
      rct->updateHistory(globalObs);
      ect->updateHistory(globalObs);

      rct->update(rHistory[traj][step + i] ? true : false);
      ect->update(endHistory[traj][step + i] ? true : false);
   }
}

void ConvolutionalBinaryCTS::sample(int act, vector<int>& sampled, bool& reward, bool& endTraj)
{   
   sample(actHistory.size() - 1, actHistory.back().size(), act, sampled, reward, endTraj);
}

void ConvolutionalBinaryCTS::sample(int traj, int step, int action, vector<int>& sampled, bool& reward, bool& endTraj)
{
   sampled.resize(width*height);

   vector<bit_t> act(bitsPerAction);
   encode(action, act);
   for(int p = 0; p < width*height; p++)
   {
      setUpContext(p, traj, step);  
      ct->updateHistory(act);
      bit_t s = ct->genRandomSymbol(uniform);
      sampled[p] = s ? 1 : 0;
   }      

   vector<bit_t> globalObs((bitsPerPixel - 1)*width*height);
   setUpContext(traj, step);
   rct->updateHistory(act);
   ect->updateHistory(act);
   encode(sampled, globalObs);
   rct->updateHistory(globalObs);
   ect->updateHistory(globalObs);
   bit_t s;
   s = rct->genRandomSymbol(uniform);
   reward = s;
   s = ect->genRandomSymbol(uniform);
   endTraj = s;
}

double ConvolutionalBinaryCTS::predict(int act, const vector<int>& obs) const
{
   double prediction = 1;
   vector<bit_t> action(bitsPerAction);
   encode(act, action);
   for(int p = 0; p < width*height; p++)
   {
      setUpContext(p, actHistory.size() - 1, actHistory.back().size());
      ct->updateHistory(action);
      bit_t symb = obs[p] ? true : false;
      double prob = ct->prob(symb);
      prediction *= prob;
   }

   return prediction;
}

double ConvolutionalBinaryCTS::predictR(int act, const vector<int>& obs, int reward) const
{
   vector<bit_t> action(bitsPerAction);
   vector<bit_t> globalObs((bitsPerPixel - 1)*width*height);
   encode(act, action);
   encode(obs, globalObs);
   setUpContext(actHistory.size() - 1, actHistory.back().size());
   rct->updateHistory(action);
   rct->updateHistory(globalObs);

   double prediction = rct->prob(reward ? true : false);

   return prediction;
}

double ConvolutionalBinaryCTS::predictEnd(int act, const vector<int>& obs, bool end) const
{
   vector<bit_t> action(bitsPerAction);
   vector<bit_t> globalObs((bitsPerPixel - 1)*width*height);
   encode(act, action);
   encode(obs, globalObs);
   setUpContext(actHistory.size() - 1, actHistory.back().size());
   ect->updateHistory(action);
   ect->updateHistory(globalObs);

   double prediction = ect->prob(end ? true : false);

   return prediction;
}

void ConvolutionalBinaryCTS::takeAction(int act, vector<int>& obs, int& reward, bool& endEpisode)
{
   bool r;
   sample(act, obs, r, endEpisode);
   reward = r;
   update(act, obs, reward, endEpisode, false);
}

void ConvolutionalBinaryCTS::takeAction(int act, int& reward, bool& endEpisode)
{
   vector<int> obs;
   bool r;
   sample(act, obs, r, endEpisode);
   reward = r;
   update(act, obs, reward, endEpisode, false);
}

void ConvolutionalBinaryCTS::saveState()
{
   savedNumTraj = actHistory.size();
   savedHistoryLength = actHistory.back().size();   
}

void ConvolutionalBinaryCTS::retrieveState()
{
   actHistory.resize(savedNumTraj);
   actHistory.back().resize(savedHistoryLength);
   obsHistory.resize(savedNumTraj);
   obsHistory.back().resize(savedHistoryLength);
   rHistory.resize(savedNumTraj);
   rHistory.back().resize(savedHistoryLength);
   endHistory.resize(savedNumTraj);
   endHistory.back().resize(savedHistoryLength);
}

